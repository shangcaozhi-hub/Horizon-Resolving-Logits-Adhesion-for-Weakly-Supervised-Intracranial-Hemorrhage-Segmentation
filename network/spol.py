import torch
from torch import nn
import torch.nn.functional as F
from backbone import resnet

class Net(nn.Module):
    def __init__(self, num_cls=1):
        super(Net, self).__init__()
        self.num_cls = num_cls

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

        self.drop_out = torch.nn.Dropout2d(0.1)

        self.classifier = nn.Conv2d(128, num_cls, kernel_size=1, stride=1, padding=0)
        # self.classifier2 = nn.Conv2d(128, num_cls, kernel_size=1, stride=1, padding=0)
        # ---- 初始化 ----
        # self._init_weights()

    def _init_weights(self):
        # 1) 需要 Kaiming 初始化的中间卷积
        intermediate_convs = [
            self.fc51, self.fc52,
            self.fc41, self.fc42,
            self.fc31, self.fc32,
            self.linear5, self.linear4, self.linear3,
        ]
        for conv in intermediate_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # 2) BN：weight=1, bias=0, running_mean=0, running_var=1
        bns = [
            self.bn51, self.bn41, self.bn31,
            self.bn5, self.bn4, self.bn3,
        ]
        for bn in bns:
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            if bn.running_mean is not None:
                nn.init.zeros_(bn.running_mean)
            if bn.running_var is not None:
                nn.init.ones_(bn.running_var)

        # 3) 分类头：小初始化，避免初始 logits 爆炸导致 BCE NaN
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        N, C, H, W = x.size()

        # forward
        out1 = self.stage0(x)
        out2 = self.stage1(out1)
        out3 = self.stage2(out2)
        out4 = self.stage3(out3)
        out5 = self.stage4(out4)

        out5a = F.relu(self.bn51(self.fc51(self.drop_out(out5.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        out4a = F.relu(self.bn41(self.fc41(self.drop_out(out4.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        out3a = F.relu(self.bn31(self.fc31(self.drop_out(out3.mean(dim=(2, 3), keepdim=True)))), inplace=True)
        vector = out5a * out4a * out3a
        out5 = torch.sigmoid(self.fc52(vector)) * out5
        out4 = torch.sigmoid(self.fc42(vector)) * out4
        out3 = torch.sigmoid(self.fc32(vector)) * out3

        out3 = F.relu(self.bn3(self.linear3(out3)))
        out4 = F.relu(self.bn4(self.linear4(out4)))
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out5 = F.relu(self.bn5(self.linear5(out5)))
        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out = out5 * out4 * out3

        # pred = self.classifier(out5)
        cams = self.classifier(out)

        # scores1 = F.adaptive_max_pool2d(pred, (1, 1))
        scores2 = F.adaptive_max_pool2d(cams, (1, 1))

        return scores2, cams
