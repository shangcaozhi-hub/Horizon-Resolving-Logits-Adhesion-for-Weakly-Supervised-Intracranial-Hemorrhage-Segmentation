import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from backbone import resnet
from network.sinkhorn import distributed_sinkhorn
from network.GMM import GaussianMixture

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class clusterSupportBank(nn.Module):
    def __init__(self, feat_dim, bank_size, num_prototypes):
        super(clusterSupportBank, self).__init__()
        self.feat_dim = feat_dim
        self.bank_size = bank_size
        self.num_prototypes = num_prototypes
        self.gamma = 0.99

        self.fore_pointer = 0

        # 初始化支持库,大小为(num_classes, bank_size, feat_dim)
        self.fore_support_bank = nn.Parameter(torch.zeros(bank_size, feat_dim), requires_grad=False)
        self.back_support_bank = nn.Parameter(torch.zeros(bank_size, feat_dim), requires_grad=False)

        self.context_prototypes = nn.Parameter(torch.zeros(2, self.num_prototypes, feat_dim), requires_grad=True)

        trunc_normal_(self.context_prototypes, std=0.02)
        trunc_normal_(self.fore_support_bank, std=0.02)
        trunc_normal_(self.back_support_bank, std=0.02)

    @torch.no_grad()
    def online_clustering(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        '''cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity'''
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.context_prototypes.data.clone()
        for k in range(2):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototypes)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototypes * k)

        self.context_prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

    @torch.no_grad()
    def update_bank(self, fore_inpro, back_inpro):

        if fore_inpro.size(0) != 0:
            temp = self.fore_support_bank[:-fore_inpro.size(0), :].to(fore_inpro.device)
            self.fore_support_bank = nn.Parameter(torch.cat([fore_inpro, temp], dim=0), requires_grad=False)

        temp = self.back_support_bank[:-back_inpro.size(0), :].to(back_inpro.device)
        self.back_support_bank = nn.Parameter(torch.cat([back_inpro, temp], dim=0), requires_grad=False)

    @torch.no_grad()
    def update_context_protos(self):
        """
        获取候选上下文原型集合
        返回: context_protos, tensor (num_classes, bank_size, feat_dim)

        """
        self.fore_support_bank.data.copy_(self.fore_support_bank)
        self.back_support_bank.data.copy_(self.back_support_bank)

        fore_num = self.fore_support_bank.size(0)
        back_num = self.back_support_bank.size(0)
        gt_normal = torch.zeros(back_num)
        gt_abnormal = torch.ones(fore_num)
        gt_seg = torch.cat([gt_normal, gt_abnormal], dim=0).to(self.fore_support_bank.device)

        ins_proto = torch.cat([self.back_support_bank, self.fore_support_bank], dim=0)
        ins_proto = l2_normalize(ins_proto)

        self.context_prototypes.data.copy_(l2_normalize(self.context_prototypes))
        masks = torch.einsum('nd,ckd->nkc', ins_proto, self.context_prototypes)
        out_seg = torch.amax(masks, dim=1)

        self.online_clustering(ins_proto, out_seg, gt_seg, masks)


class PrototypeSupportBank(nn.Module):
    def __init__(self, feat_dim, bank_size, num_classes, num_prototypes, gamma, num_components=3):
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
                num_component = 1
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
                        gmm = GaussianMixture(self.num_components, self.feat_dim, covariance_type="full").to(
                            queue.device)
                        self.init_flag[i] = False
                    else:
                        gmm = GaussianMixture(self.num_components, self.feat_dim, covariance_type="full",
                                              mu_init=old_means.unsqueeze(0), var_init=old_covariances.unsqueeze(0)).to(
                            queue.device)

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
                    updated_covariances = self.gamma * old_covariances.squeeze(0) + (
                                1 - self.gamma) * new_covariances + cov_correction
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
            z = torch.randn(self.num_prototypes, self.feat_dim).to(
                selected_covs.device)  # shape: [num_samples, feature_dim]

            # 批量矩阵乘法: samples = means + L @ z
            # 需要调整z的形状以便进行批量矩阵乘法
            z = z.unsqueeze(2)  # shape: [num_samples, feature_dim, 1]
            transformed = torch.bmm(L, z).squeeze(2)  # shape: [num_samples, feature_dim]

            # 添加均值得到最终样本
            samples = selected_means + transformed  # shape: [num_samples, feature_dim]

            prototypes.append(samples.unsqueeze(0))

        prototypes = torch.cat(prototypes, dim=0)

        return prototypes


class SPOL_pretrain(nn.Module):
    def __init__(self, num_cls=1, num_prototype=50, gamma=0.9, threshold=None):
        super(SPOL_pretrain, self).__init__()
        self.num_cls = num_cls
        self.threshold = threshold
        self.test = False

        self.resnet50 = resnet.resnet50(pretrained=False, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.fc51 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn51 = nn.BatchNorm2d(128)
        self.fc52 = nn.Conv2d(128, 2048, kernel_size=1, stride=1, padding=0)
        self.fc41 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn41 = nn.BatchNorm2d(128)
        self.fc42 = nn.Conv2d(128, 1024, kernel_size=1, stride=1, padding=0)
        self.fc31 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn31 = nn.BatchNorm2d(128)
        self.fc32 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.linear5 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.linear4 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.linear3 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)

        self.classifier = nn.Conv2d(128, num_cls, kernel_size=1, stride=1, padding=0)

        self.side1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.side2 = torch.nn.Conv2d(1024, 64, 1, bias=False)
        self.side3 = torch.nn.Conv2d(2048, 64, 1, bias=False)
        self.side4 = torch.nn.Conv2d(128, 125, 1, bias=False)
        self.proj_head = torch.nn.Linear(320, 320, bias=False)

        # self.mlp = Mlp(387, 387, drop=0.1)
        # self.ln = nn.LayerNorm(387)

        if threshold == None:
            self.PrototypeSupportBank_05 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
            self.PrototypeSupportBank_06 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
            self.PrototypeSupportBank_07 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
            self.PrototypeSupportBank_08 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
            self.PrototypeSupportBank_09 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
            self.clusterSupportBank = clusterSupportBank(320, 1000, 50)
        elif threshold == 0.9:
            self.PrototypeSupportBank_09 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.8:
            self.PrototypeSupportBank_08 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.7:
            self.PrototypeSupportBank_07 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.6:
            self.PrototypeSupportBank_06 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.5:
            self.PrototypeSupportBank_05 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.99:
            self.PrototypeSupportBank_099 = PrototypeSupportBank(320, 1000, self.num_cls + 1, num_prototype, gamma)
        elif threshold == 0.1:
            self.clusterSupportBank = clusterSupportBank(320, 1000, 50)

        self.drop_out = torch.nn.Dropout2d(0.1)

        self.feat_norm = nn.LayerNorm(320)
        for param in self.feat_norm.parameters():
            param.requires_grad = False

    def forward(self, x, label=None, skull_mask=None):
        out1 = self.stage0(x)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out4 = self.stage3(out3)
        out5 = self.stage4(out4)

        side1 = self.side1(out3.detach())
        side2 = self.side2(out4.detach())
        side3 = self.side3(out5.detach())

        out5a = F.relu(self.bn51(self.fc51(self.drop_out(out5.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        out4a = F.relu(self.bn41(self.fc41(self.drop_out(out4.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        out3a = F.relu(self.bn31(self.fc31(self.drop_out(out3.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        vector = out5a * out4a * out3a
        out5_ = torch.sigmoid(self.fc52(vector)) * out5
        out4_ = torch.sigmoid(self.fc42(vector)) * out4
        out3_ = torch.sigmoid(self.fc32(vector)) * out3

        out3_ = F.relu(self.bn3(self.linear3(out3_)), inplace=True)
        out4_ = F.relu(self.bn4(self.linear4(out4_)), inplace=True)
        out4_ = F.interpolate(out4_, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out5_ = F.relu(self.bn5(self.linear5(out5_)), inplace=True)
        out5_ = F.interpolate(out5_, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out = out5_ * out4_ * out3_

        side4 = self.side4(out)

        x_ = F.interpolate(x, size=out3.size()[2:], mode='bilinear', align_corners=True)

        side1 = F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), size=out3.size()[2:],
                              mode='bilinear', align_corners=True)
        side2 = F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), size=out3.size()[2:],
                              mode='bilinear', align_corners=True)
        side3 = F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), size=out3.size()[2:],
                              mode='bilinear', align_corners=True)
        side4 = F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), size=out3.size()[2:],
                              mode='bilinear', align_corners=True)

        # scores = F.adaptive_max_pool2d(self.classifier(out), (1, 1))

        cam = self.classifier(out)

        # norm_cam = torch.relu(cam)
        # norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        #
        # cam = self.classifier((1 - norm_cam) * out)
        scores = F.adaptive_max_pool2d(cam, (1, 1))

        hie_fea = torch.cat([x_, side1, side2, side3, side4], dim=1)
        n, _, h, w = hie_fea.size()
        hie_fea = rearrange(hie_fea, "b d h w-> b (h w) d")
        hie_fea = self.feat_norm(F.relu(self.proj_head(hie_fea)))

        if self.test == False:
            if self.threshold == None:

                norm_cam = F.relu(cam)
                norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)  # b c h w
                norm_cam = F.interpolate(norm_cam, (h, w), mode='bilinear', align_corners=True)

                fore_mask = norm_cam > 0.1
                back_mask = (1 - norm_cam) > 0.1  # b 1 h w
                valid_mask = (fore_mask.sum([2, 3]) > 0).squeeze()  # b 1

                fore_mask = rearrange(fore_mask, "b d h w-> b (h w) d")
                back_mask = rearrange(back_mask, "b d h w-> b (h w) d")
                fore_inpro = (fore_mask * hie_fea).sum(dim=1) / (fore_mask.sum(dim=1) + 1e-6)
                fore_inpro = fore_inpro[valid_mask]
                back_inpro = (back_mask * hie_fea).sum(dim=1) / (back_mask.sum(dim=1) + 1e-6)

                self.clusterSupportBank.update_bank(fore_inpro, back_inpro)
                self.clusterSupportBank.update_context_protos()

                if skull_mask != None:
                    skull_mask = (F.interpolate(skull_mask.squeeze(1), (h, w), mode='bilinear',
                                                align_corners=True) > 0)

                if label != None:
                    norm_cam = norm_cam * label.unsqueeze(-1).unsqueeze(-1).float()
                    valid_index = torch.nonzero(torch.eq(label.squeeze(), 1)).view(-1)

                    if not valid_index.any():  # 检查是否全为False
                        random_idx = torch.randint(norm_cam.size(0), (1,), device=norm_cam.device)
                        # 创建副本并修改
                        valid_index[random_idx] = True
                        print(valid_index)
                    norm_cam = norm_cam[valid_index]
                    hie_fea = hie_fea[valid_index]
                    skull_mask = skull_mask[valid_index.cpu()]

                fore_mask_09 = norm_cam > 0.9
                fore_mask_08 = norm_cam > 0.8
                fore_mask_07 = norm_cam > 0.7
                fore_mask_06 = norm_cam > 0.6
                fore_mask_05 = norm_cam > 0.5

                uncertain_mask = norm_cam > 0

                pseudo_label_09 = torch.zeros_like(norm_cam)
                pseudo_label_08 = torch.zeros_like(norm_cam)
                pseudo_label_07 = torch.zeros_like(norm_cam)
                pseudo_label_06 = torch.zeros_like(norm_cam)
                pseudo_label_05 = torch.zeros_like(norm_cam)

                if skull_mask != None:
                    pseudo_label_05[skull_mask == False] = -2
                    pseudo_label_06[skull_mask == False] = -2
                    pseudo_label_07[skull_mask == False] = -2
                    pseudo_label_08[skull_mask == False] = -2
                    pseudo_label_09[skull_mask == False] = -2

                pseudo_label_05[uncertain_mask == True] = -1
                pseudo_label_06[uncertain_mask == True] = -1
                pseudo_label_07[uncertain_mask == True] = -1
                pseudo_label_08[uncertain_mask == True] = -1
                pseudo_label_09[uncertain_mask == True] = -1

                pseudo_label_09[fore_mask_09 == True] = 1
                pseudo_label_08[fore_mask_08 == True] = 1
                pseudo_label_07[fore_mask_07 == True] = 1
                pseudo_label_06[fore_mask_06 == True] = 1
                pseudo_label_05[fore_mask_05 == True] = 1

                pseudo_label_09 = rearrange(pseudo_label_09, 'b c h w -> (b h w c)')
                pseudo_label_08 = rearrange(pseudo_label_08, 'b c h w -> (b h w c)')
                pseudo_label_07 = rearrange(pseudo_label_07, 'b c h w -> (b h w c)')
                pseudo_label_06 = rearrange(pseudo_label_06, 'b c h w -> (b h w c)')
                pseudo_label_05 = rearrange(pseudo_label_05, 'b c h w -> (b h w c)')
                self.PrototypeSupportBank_09.update_bank(pseudo_label_09, hie_fea.clone().detach())
                self.PrototypeSupportBank_08.update_bank(pseudo_label_08, hie_fea.clone().detach())
                self.PrototypeSupportBank_07.update_bank(pseudo_label_07, hie_fea.clone().detach())
                self.PrototypeSupportBank_06.update_bank(pseudo_label_06, hie_fea.clone().detach())
                self.PrototypeSupportBank_05.update_bank(pseudo_label_05, hie_fea.clone().detach())

                return scores, cam

            elif self.threshold == 0.1:

                norm_cam = F.relu(cam)
                norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)  # b c h w
                norm_cam = F.interpolate(norm_cam, (h, w), mode='bilinear', align_corners=True)

                fore_mask = norm_cam > 0.1
                back_mask = (1 - norm_cam) > 0.1  # b 1 h w
                valid_mask = (fore_mask.sum([2, 3]) > 0).squeeze()  # b 1

                fore_mask = rearrange(fore_mask, "b d h w-> b (h w) d")
                back_mask = rearrange(back_mask, "b d h w-> b (h w) d")
                fore_inpro = (fore_mask * hie_fea).sum(dim=1) / (fore_mask.sum(dim=1) + 1e-6)
                fore_inpro = fore_inpro[valid_mask]
                back_inpro = (back_mask * hie_fea).sum(dim=1) / (back_mask.sum(dim=1) + 1e-6)

                return scores, cam, hie_fea, fore_inpro, back_inpro
            else:
                return scores, cam, hie_fea, out
        else:
            return scores, cam, hie_fea, out
