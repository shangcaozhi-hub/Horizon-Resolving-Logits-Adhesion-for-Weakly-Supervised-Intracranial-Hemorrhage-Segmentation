import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from network.spol_features import SPOL_pretrain

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


class Plug_play(nn.Module):
    def __init__(self, num_cls=1, num_prototype=50, gamma=0.9, threshold=0.8):
        super(Plug_play, self).__init__()
        self.num_cls = num_cls
        self.num_prototype = num_prototype
        self.gamma = gamma
        self.threshold = threshold

        self.backbone = SPOL_pretrain(num_cls, num_prototype, gamma, threshold)
        self.backbone.test = True

        self.mask_norm = nn.LayerNorm(num_cls + 1)

        # self.seg_head = torch.nn.Conv2d(128, 1, 1, bias=False)
        # self.proj_head = torch.nn.Linear(320, 320, bias=False)

        if threshold == 0.9:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_09
        elif threshold == 0.8:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_08
        elif threshold == 0.7:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_07
        elif threshold == 0.6:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_06
        elif threshold == 0.5:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_05
        elif threshold == 0.99:
            self.PrototypeSupportBank = self.backbone.PrototypeSupportBank_099

        self.n_components = self.PrototypeSupportBank.num_components


    def cam_phase(self, x, label=None, skull_mask=None):

        scores, cam, hie_fea, out = self.backbone(x)

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

        return scores, cam, hie_fea, fore_mask, uncertain_mask

    def proto_phase(self, hie_fea, cam):

        n, _, h, w = cam.size()

        sampled_protos = self.PrototypeSupportBank.sample_prototypes().to(hie_fea.device).detach()  # c k d
        random_proto_back = self.PrototypeSupportBank.gmm_means_0.unsqueeze(0)
        global_proto_fore = self.PrototypeSupportBank.gmm_means_1.unsqueeze(0).repeat(1, self.n_components, 1)
        global_proto = torch.cat([random_proto_back, global_proto_fore], dim=0)

        # proto_fea = rearrange(l2_normalize(self.proj_head(hie_fea)), 'b n d -> (b n) d')
        # sampled_protos = l2_normalize(self.proj_head(sampled_protos))

        proto_fea = l2_normalize(hie_fea).reshape(n * h * w, hie_fea.size(-1))
        sampled_protos = l2_normalize(sampled_protos)
        global_proto = l2_normalize(global_proto)

        proto_logits = torch.einsum('nd, ckd-> nck', proto_fea, sampled_protos)
        global_logits = torch.einsum('nd, ckd-> nck', proto_fea, global_proto) # 保证稳定性
        proto_logits = torch.cat([proto_logits, global_logits], dim=-1)  # .mean(-1).unsqueeze(-1)
        # proto_logits = torch.einsum('nd, ckd-> nck', proto_fea, sampled_protos.mean(1).unsqueeze(1))

        # sim_back = proto_logits[:, :1, -self.n_components:].amax(-1)
        sim_back = proto_logits[:, :1, :].amax(-1)
        sim_fore = proto_logits[:, 1:, :].amax(-1)
        sim_fore_mean = proto_logits[:, 1:, -1]

        sim_max = torch.cat([sim_back, sim_fore], dim=1)

        # p_logits = self.mask_norm(sim_mean)
        sim_max = rearrange(sim_max, '(b h w) c -> b c h w', b=n, h=h, w=w)
        # p_logits = rearrange(sim_fore_mean, '(b h w) c -> b c h w', b=n, h=h, w=w)
        all_logits = rearrange(proto_logits, '(b h w) c k -> b c k h w', b=n, h=h, w=w)

        proto_fore_mask = (sim_max[:, 1:, :, :] - sim_max[:, :1, :, :]) > 0

        return proto_logits, sim_max, all_logits, proto_fore_mask

    def forward(self, x):

        scores, cam, hie_fea, fore_mask, _ = self.cam_phase(x)

        n, _, h, w = cam.size()
        '''proto_fea = l2_normalize(hie_fea).reshape(n * h * w, hie_fea.size(-1))
        sampled_protos = l2_normalize(sampled_protos)

        proto_logits = torch.einsum('nd, ckd-> nck', proto_fea, sampled_protos)

        sim_back = proto_logits[:, :1, :].amax(-1)
        sim_fore = proto_logits[:, 1:, :].amax(-1)

        sim_max = torch.cat([sim_back, sim_fore], dim=1)

        sim_max = rearrange(sim_max, '(b h w) c -> b c h w', b=n, h=h, w=w)
        all_logits = rearrange(proto_logits, '(b h w) c k -> b c k h w', b=n, h=h, w=w)'''

        proto_logits, sim, global_logits, _ = self.proto_phase(hie_fea, cam)
        # sim2 = rearrange(sim, '(b h w) c -> b c h w', b=n, h=h, w=w)

        return scores, cam, sim, global_logits
