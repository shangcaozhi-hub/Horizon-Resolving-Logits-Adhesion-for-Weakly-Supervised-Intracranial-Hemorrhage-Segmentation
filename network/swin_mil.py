import torch
from torch import nn
import torch.nn.functional as F
from backbone.SwinTransformer import SwinTransformer

class SwinMIL(nn.Module):

    def __init__(self,  num_classes=1, ):
        super(SwinMIL, self).__init__()
        self.num_classes = num_classes

        self.backbone = SwinTransformer(depths=(2, 2, 6, 2), out_indices=(0, 1, 2))

        self.decoder = nn.Conv2d(672, 1, kernel_size=1, stride=1, padding=0)


    def cam_phase(self, x):

        x1, x2, x3 = self.backbone(x)

        _, _, h, w = x2.size()
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=True)

        out = torch.cat([x1, x2, x3], dim=1)
        cam = self.decoder(out)

        score = F.adaptive_max_pool2d(cam, (1, 1)).view(-1)  # F.adaptive_max_pool2d(logits, (1, 1)).view(-1)

        return score, cam

    def forward(self, x):

        scores, cam = self.cam_phase(x)

        return scores, cam
