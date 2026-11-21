import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from backbone.SwinTransformer import SwinTransformer
from network.GMM import GaussianMixture

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class PrototypeSupportBank(nn.Module):
    def __init__(self, feat_dim, bank_size, num_classes, num_prototypes, gamma, num_components=5, foreground_components=5):
        super(PrototypeSupportBank, self).__init__()
        self.feat_dim = feat_dim
        self.bank_size = bank_size
        self.gamma = gamma
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.num_components = num_components
        self.flag = []

        # 初始化支持库,大小为(num_classes, bank_size, feat_dim)

        self.init_flag = []

        for i in range(0, self.num_classes):
            self.flag.append(0)
            self.init_flag.append(True)

            if i == 1:
                num_component = foreground_components
            else:
                num_component = num_components

            self.register_buffer("queue" + str(i), torch.zeros(self.bank_size, self.feat_dim))
            self.register_buffer("queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))

            # 为每个类初始化GMM参数
            # weights: 混合权重
            self.register_buffer(f"gmm_weights_{i}",
                                 torch.ones(num_component) / num_component)

            # means: 每个组件的均值
            self.register_buffer(f"gmm_means_{i}",
                                 torch.zeros(num_component, self.feat_dim))

            # covariances: 每个组件的协方差矩阵
            self.register_buffer(f"gmm_covariances_{i}",
                                 torch.stack([torch.eye(self.feat_dim) for _ in range(num_component)]))

    @torch.no_grad()
    def update_bank(self, pseudo_label, hie_fea):

        hie_fea = rearrange(hie_fea, "b n d-> (b n) d")

        ins_proto = []

        for i in range(self.num_classes):
            # 如果是背景，随机选一个patch作为 ins_proto
            indices = (pseudo_label == i).nonzero()

            # 初始化选取的索引
            # temp = torch.zeros(hie_fea.size(0), hie_fea.size(-1)).to(hie_fea.device)

            # 对每个批次选取一个索引
            if indices.size(0) > 0:
                if i == 0:
                    num_to_select = max(int(indices.size(0) * 0.01), 1)
                    random_indices = torch.randperm(indices.size(0))[:num_to_select]
                    random_indices = indices[random_indices]
                    samples = hie_fea[random_indices, :].squeeze(1)

                    black_indices = (pseudo_label == -2).nonzero()
                    random_black = torch.randperm(black_indices.size(0))[:num_to_select]
                    random_black = black_indices[random_black]
                    black_samples = hie_fea[random_black, :].squeeze(1)

                    samples = torch.cat([samples, black_samples], dim=0)
                else:
                    num_to_select = max(int(indices.size(0) * 0.5), 1)
                    random_indices = torch.randperm(indices.size(0))[:num_to_select]
                    random_indices = indices[random_indices]
                    samples = hie_fea[random_indices, :].squeeze(1)
            else:
                samples = getattr(self, "queue" + str(i))[0, :].unsqueeze(0)

            # temp = torch.cat(temp, dim=0)
            ins_proto.append(samples)

        for idx in range(self.num_classes):

            ins_proto_i = ins_proto[idx]

            num_ins = ins_proto_i.size(0)

            queue_i = getattr(self, "queue" + str(idx))
            queue_ptr_i = getattr(self, "queue_ptr" + str(idx))

            if num_ins > self.bank_size:
                num_ins = self.bank_size
                ins_proto_i = ins_proto_i[:num_ins, :]
                queue_ptr_i[0] = self.bank_size
            else:
                queue_ptr_i[0] = int(queue_ptr_i) + num_ins

            queue_i = torch.cat([ins_proto_i, queue_i[:-num_ins, :]], dim=0)

            self.register_buffer("queue" + str(idx), queue_i)
            self.register_buffer("queue_ptr" + str(idx), queue_ptr_i)

        self.update_statistics()

    @torch.no_grad()
    def update_statistics(self):

        queue_ptr_1 = getattr(self, "queue_ptr" + str(1))
        if queue_ptr_1 > self.bank_size:

            for i in range(self.num_classes):

                self.flag[i] += 1
                # 指针清0

                queue = getattr(self, "queue" + str(i))

                # 使用sklearn的GaussianMixture进行拟合
                # 获取旧参数
                old_weights = getattr(self, f"gmm_weights_{i}")
                old_means = getattr(self, f"gmm_means_{i}")
                old_covariances = getattr(self, f"gmm_covariances_{i}")

                # 初始化或获取已有GMM
                if i == 0:
                    if self.init_flag[i]:
                        gmm = GaussianMixture(self.num_components, self.feat_dim, covariance_type="full").to(queue.device)
                        self.init_flag[i] = False
                    else:
                        gmm = GaussianMixture(self.num_components, self.feat_dim, covariance_type="full", mu_init=old_means.unsqueeze(0), var_init=old_covariances.unsqueeze(0)).to(queue.device)

                    gmm.fit(queue)  # 执行EM拟合

                    # 获取更新后参数
                    new_weights = gmm.pi.data.squeeze(0).squeeze(-1)
                    new_means = gmm.mu.data.squeeze(0)
                    new_covariances = gmm.var.data.squeeze(0)

                    delta_mean = (old_means - new_means).unsqueeze(1)
                    cov_correction = self.gamma * (1 - self.gamma) * (delta_mean.permute(0, 2, 1) @ delta_mean)  # k n n

                    # 使用动量更新参数
                    updated_weights = self.gamma * old_weights + (1 - self.gamma) * new_weights
                    updated_means = self.gamma * old_means + (1 - self.gamma) * new_means
                    updated_covariances = self.gamma * old_covariances + (
                                1 - self.gamma) * new_covariances + cov_correction

                    # 重新归一化权重
                    updated_weights = updated_weights / updated_weights.sum()

                    # 确保协方差矩阵是对称且正定的
                    for j in range(self.num_components):
                        cov = updated_covariances[j]
                        max_elem = torch.max(torch.diag(cov))  # 确保对称性
                        cov += 1e-6 * max_elem * torch.eye(cov.size(-1), device=cov.device)
                        updated_covariances[j] = cov
                else:
                    new_means = torch.mean(queue, dim=0, keepdim=True)

                    centered = queue - new_means

                    # 计算协方差矩阵
                    new_covariances = torch.matmul(centered.T, centered) / (self.bank_size - 1)

                    delta_mean = old_means - new_means
                    cov_correction = self.gamma * (1 - self.gamma) * (delta_mean.T @ delta_mean)

                    # 更新协方差，加入自适应正则化
                    updated_covariances = self.gamma * old_covariances.squeeze(0) + (1 - self.gamma) * new_covariances + cov_correction
                    max_elem = torch.max(torch.diag(updated_covariances))  # 自适应正则化系数
                    updated_covariances += 1e-6 * max_elem * torch.eye(new_means.size(-1), device=new_means.device)
                    updated_covariances = updated_covariances.unsqueeze(0)

                    updated_weights = old_weights
                    updated_means = self.gamma * old_means + (1 - self.gamma) * new_means

                # 注册更新后的参数
                self.register_buffer(f"gmm_weights_{i}", updated_weights)
                self.register_buffer(f"gmm_means_{i}", updated_means)
                self.register_buffer(f"gmm_covariances_{i}", updated_covariances)

            queue_ptr_1[0] = 0
            self.register_buffer("queue_ptr" + str(0), queue_ptr_1)

    @torch.no_grad()
    def sample_prototypes(self):

        prototypes = []

        for i in range(self.num_classes):

            weights = getattr(self, f"gmm_weights_{i}")
            means = getattr(self, f"gmm_means_{i}")
            covariances = getattr(self, f"gmm_covariances_{i}")

            # 根据权重随机选择组件
            selected_components = torch.multinomial(weights, self.num_prototypes, replacement=True)

            # 获取选定组件的均值
            selected_means = means[selected_components]  # shape: [num_samples, feature_dim]

            # 获取选定组件的协方差矩阵
            selected_covs = covariances[selected_components]  # shape: [num_samples, feature_dim, feature_dim]

            # 使用批量Cholesky分解
            L = torch.linalg.cholesky(selected_covs)  # shape: [num_samples, feature_dim, feature_dim]

            # 生成标准正态随机向量
            z = torch.randn(self.num_prototypes, self.feat_dim).to(selected_covs.device)  # shape: [num_samples, feature_dim]

            # 批量矩阵乘法: samples = means + L @ z
            # 需要调整z的形状以便进行批量矩阵乘法
            z = z.unsqueeze(2)  # shape: [num_samples, feature_dim, 1]
            transformed = torch.bmm(L, z).squeeze(2)  # shape: [num_samples, feature_dim]

            # 添加均值得到最终样本
            samples = selected_means + transformed  # shape: [num_samples, feature_dim]

            prototypes.append(samples.unsqueeze(0))

        prototypes = torch.cat(prototypes, dim=0)

        return prototypes


class PrototypeSwinTransformer(nn.Module):

    def __init__(self, num_prototype=50, num_classes=1, gamma=0.9, threshold=0.8, auxiliary_head=True, bank_size=1000, components=5, foreground_components=5):
        super(PrototypeSwinTransformer, self).__init__()
        self.num_prototype = num_prototype
        self.num_classes = num_classes
        self.gamma = gamma
        self.threshold = threshold

        self.backbone = SwinTransformer(depths=(2, 2, 6, 2), out_indices=(0, 1, 2))

        # self.decoder1 = nn.Conv2d(96, 1, kernel_size=1, stride=1, padding=0)
        # self.decoder2 = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        # self.decoder3 = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.decoder = nn.Conv2d(672, 1, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Conv2d(384, 1, kernel_size=1) if auxiliary_head else None

        #  384 + 96 + 192 18 27 288
        # self.bn = nn.BatchNorm2d(128)

        self.side_1 = torch.nn.Conv2d(96, 64, 1, bias=False)
        self.side_2 = torch.nn.Conv2d(192, 64, 1, bias=False)
        self.side_3 = torch.nn.Conv2d(384, 125, 1, bias=False)
        self.proj_head = torch.nn.Linear(256, 256, bias=False)

        self.feat_norm = nn.LayerNorm(256)

        self.PrototypeSupportBank = PrototypeSupportBank(256, bank_size, num_classes+1, num_prototype, gamma, num_components=components, foreground_components=foreground_components)
        self.n_components = self.PrototypeSupportBank.num_components
    

    def cam_phase(self, x, label=None, skull_mask=None):

        x1, x2, x3 = self.backbone(x)

        pred = self.classifier(x3) if self.classifier is not None else None

        _, _, h, w = x2.size()
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=True)

        out = torch.cat([x1, x2, x3], dim=1)
        cam = self.decoder(out)

        # _, _, h, w = x2.size()
        # cam1 = F.interpolate(cam1, size=(h, w), mode="bilinear", align_corners=True)
        # cam3 = F.interpolate(cam3, size=(h, w), mode="bilinear", align_corners=True)

        # cam = self.w[0] * cam1 + self.w[1] * cam2 + self.w[2] * cam3
        # logits = self.w[0] * torch.sigmoid(cam1) + self.w[1] * torch.sigmoid(cam2) + self.w[2] * torch.sigmoid(cam3)

        side1 = self.side_1(x1.detach())
        side2 = self.side_2(x2.detach())
        side3 = self.side_3(x3)

        x_ = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        side1 = F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), size=x2.size()[2:],
                              mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), size=x2.size()[2:],
                              mode='bilinear', align_corners=True)
        side3 = F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), size=x2.size()[2:],
                             mode='bilinear', align_corners=True)

        hie_fea = torch.cat([x_, side1, side2, side3], dim=1)
        n, _, h, w = hie_fea.size()
        hie_fea = rearrange(hie_fea, "b d h w-> b (h w) d")
        hie_fea = self.feat_norm(F.relu(self.proj_head(hie_fea), inplace=True))

        score = F.adaptive_max_pool2d(cam, (1, 1)).view(-1) # F.adaptive_max_pool2d(logits, (1, 1)).view(-1)
        auxiliary_scores = [F.adaptive_max_pool2d(pred, (1, 1)).view(-1)] if pred is not None else []

        b, n, d = hie_fea.size()
        h = w = int(n ** 0.5)

        with torch.no_grad():
            norm_cam = F.relu(cam)
            norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)  # b c h w
            norm_cam = F.interpolate(norm_cam, (h, w), mode='bilinear', align_corners=True)

            pseudo_label = torch.zeros_like(norm_cam)

            if skull_mask != None:
                skull_mask = (F.interpolate(skull_mask.squeeze(1), (h, w), mode='bilinear',
                                            align_corners=True) > 0)

            if label is not None and label.any():
                norm_cam = norm_cam * label.unsqueeze(-1).unsqueeze(-1).float()
                valid_index = torch.nonzero(torch.eq(label.reshape(-1), 1)).view(-1)

                norm_cam_ = norm_cam[valid_index]
                hie_fea_ = hie_fea[valid_index, :, :]
                skull_mask_ = skull_mask[valid_index.cpu()]

                fore_mask = norm_cam_ > self.threshold
                uncertain_mask = norm_cam_ > 0

                pseudo_label = pseudo_label[valid_index]
                pseudo_label[uncertain_mask == True] = -1
                pseudo_label[fore_mask == True] = 1
                pseudo_label[skull_mask_ == False] = -2

                self.PrototypeSupportBank.update_bank(pseudo_label.view(-1), hie_fea_)

        fore_mask = norm_cam > self.threshold
        uncertain_mask = norm_cam > 0
        scores = [score, *auxiliary_scores]

        return scores, cam, hie_fea, fore_mask, uncertain_mask

    def proto_phase(self, hie_fea, cam):

        n, _, h, w = cam.size()

        sampled_protos = self.PrototypeSupportBank.sample_prototypes().to(hie_fea.device).detach()  # c k d
        random_proto_back = self.PrototypeSupportBank.gmm_means_0.unsqueeze(0)
        global_proto_fore = self.PrototypeSupportBank.gmm_means_1.unsqueeze(0).expand(1, self.n_components, -1)
        global_proto = torch.cat([random_proto_back, global_proto_fore], dim=0)
        # proto_fea = rearrange(l2_normalize(self.proj_head(hie_fea)), 'b n d -> (b n) d')
        # sampled_protos = l2_normalize(self.proj_head(sampled_protos))

        proto_fea = l2_normalize(hie_fea).reshape(n * h * w, hie_fea.size(-1))
        sampled_protos = l2_normalize(sampled_protos)
        global_proto = l2_normalize(global_proto)

        proto_logits = torch.einsum('nd, ckd-> nck', proto_fea, sampled_protos)
        global_logits = torch.einsum('nd, ckd-> nck', proto_fea, global_proto)
        proto_logits = torch.cat([proto_logits, global_logits], dim=-1)

        # sim_back = proto_logits[:, :1, -self.n_components:].amax(-1)
        sim_back = proto_logits[:, :1, :].amax(-1)
        sim_fore = proto_logits[:, 1:, :].amax(-1)
        sim_fore_mean = proto_logits[:, 1:, -1]

        sim_max = torch.cat([sim_back, sim_fore], dim=1)

        # p_logits = self.mask_norm(sim_mean)
        sim_max = rearrange(sim_max, '(b h w) c -> b c h w', b=n, h=h, w=w)
        p_logits = rearrange(sim_fore_mean, '(b h w) c -> b c h w', b=n, h=h, w=w)

        proto_fore_mask = (sim_max[:, 1:, :, :] - sim_max[:, :1, :, :]) > 0

        return proto_logits, torch.relu(sim_max), torch.relu(p_logits), proto_fore_mask

    def forward(self, x):

        B, C, H, W = x.size()

        scores, cam, hie_fea, fore_mask, _ = self.cam_phase(x)

        H, W = H // 8, W // 8
        norm_cams = F.relu(cam)
        norm_cams = norm_cams / (F.adaptive_max_pool2d(norm_cams.detach(), (1, 1)) + 1e-5)
        norm_cams = F.interpolate(norm_cams, (H, W), mode='bilinear', align_corners=True)

        proto_logits, sim, sim2, _ = self.proto_phase(hie_fea, cam)

        return scores, cam, sim, norm_cams
