import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import math
import torch.nn.functional as F
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 -> num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer1_output = x
        x = self.layer2(x)
        layer2_output = x
        x = self.layer3(x)
        layer3_output = x
        x = self.layer4(x)
        layer4_output = x
        x = self.avgpool(x)
        return x

    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        #backbone = [self.conv1, self.bn1, self.layer1, self.layer2,self.layer3]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        #backbone = [self.conv1, self.bn1, self.layer1, self.layer2]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


def resnet18(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 ENFIEC.
    Args:
        pretrained (bool): If True, returns a ENFIEC pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    # ENFIEC.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        model.apply(weights_init)
        # pretrained_dict = load_state_dict_from_url(
        #     "https://download.pytorch.org/models/resnet18-5c106cde.pth")
        # model_dict = ENFIEC.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        # model_dict.update(pretrained_dict)
        # ENFIEC.load_state_dict(model_dict)
        # print("成功加载预训练权重")
        # ENFIEC.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def resnet34(num_classes=1000, pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    # ENFIEC.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet34-333f7ec4.pth")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("成功加载预训练权重")
    return model


def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 ENFIEC.
    Args:
        pretrained (bool): If True, returns a ENFIEC pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    # ENFIEC.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("成功加载预训练权重")
        # ENFIEC.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 ENFIEC.
    Args:
        pretrained (bool): If True, returns a ENFIEC pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    #ENFIEC.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-19c8e357.pth")
        # pretrained_dict = torch.load("F:/Pyprojects/cl/logs_new/"+"simclr_pretrained2.0_5000.pth.tar")["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("成功加载预训练权重")
        # ENFIEC.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


class Multimodel_no_text50_without_class(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50_without_class, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Linear(2048, 256),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Linear(256, 128),  # 4096+2    #4096+59
        )

    def forward(self, x, x_withCDFI):
        x = self.model(x)[0]
        x = torch.flatten(x, 1)
        x_withCDFI = self.model_CDFI(x_withCDFI)
        x_withCDFI = torch.flatten(x_withCDFI, 1)
        feature = torch.cat((x, x_withCDFI), 1)
        output = self.fc(feature)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()


class Multimodel_with_three(nn.Module):
    def __init__(self, pretrained, num_classes, benign_3, malignant_3, malignant_4a):
        super(Multimodel_with_three, self).__init__()
        self.num_classes = num_classes
        self.model_benign_3 = Multimodel_no_text50_without_class(num_classes=self.num_classes, pretrained=pretrained)
        model_dict = self.model_benign_3.state_dict()
        pretrained_dict = torch.load(benign_3)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_benign_3.load_state_dict(model_dict)
        self.model_malignant_3 = Multimodel_no_text50_without_class(num_classes=self.num_classes, pretrained=pretrained)
        model_dict = self.model_malignant_3.state_dict()
        pretrained_dict = torch.load(malignant_3)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_malignant_3.load_state_dict(model_dict)
        self.model_malignant_4a = Multimodel_no_text50_without_class(num_classes=self.num_classes,
                                                                     pretrained=pretrained)
        model_dict = self.model_malignant_4a.state_dict()
        pretrained_dict = torch.load(malignant_4a)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_malignant_4a.load_state_dict(model_dict)
        self.fc = nn.Sequential(
            nn.Linear(6144, 2048),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, x_withCDFI):
        x_benign_3 = self.model_benign_3(x, x_withCDFI)
        x_malignant_3 = self.model_malignant_3(x, x_withCDFI)
        x_malignant_4a = self.model_malignant_4a(x, x_withCDFI)
        feature = torch.cat((x_benign_3, x_malignant_3, x_malignant_4a), 1)
        output = self.fc(feature)
        return output

    def Multi_freeze_backbone(self):
        self.model_malignant_3.Multi_freeze_backbone()
        self.model_malignant_4a.Multi_freeze_backbone()
        self.model_benign_3.Multi_freeze_backbone()


class Multimodel_no_text50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes),  # 4096+2    #4096+59
            nn.ReLU(),
        )

    def forward(self, x, x_withCDFI):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x_withCDFI = self.model_CDFI(x_withCDFI)
        x_withCDFI = torch.flatten(x_withCDFI, 1)
        feature = torch.cat((x, x_withCDFI), 1)
        output = self.fc(feature)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()


class Multimodel_Multiexperts(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_Multiexperts, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)

    def forward(self, x, x_withCDFI):
        x, x_1, x_2, x_3, x_4 = self.model(x)
        #x = torch.flatten(x, 1)
        y, y_1, y_2, y_3, y_4 = self.model_CDFI(x_withCDFI)
        #x_withCDFI = torch.flatten(x_withCDFI, 1)
        #feature = torch.cat((x, x_withCDFI), 1)
        return x, x_1, x_2, x_3, x_4


def resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 ENFIEC.
    Args:
        pretrained (bool): If True, returns a ENFIEC pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    #ENFIEC.avgpool = nn.AdaptiveAvgPool2d(1)
    if pretrained:
        pretrained_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-19c8e357.pth")
        # pretrained_dict = torch.load("F:/Pyprojects/cl/logs_new/"+"simclr_pretrained2.0_5000.pth.tar")["state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("成功加载预训练权重")
        # ENFIEC.load_state_dict(load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


class Multimodel_no_text50_without_class(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50_without_class, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, x_withCDFI):
        x = self.model(x)[0]
        x = torch.flatten(x, 1)
        x_withCDFI = self.model(x_withCDFI)[0]
        x_withCDFI = torch.flatten(x_withCDFI, 1)
        feature = torch.cat((x, x_withCDFI), 1)
        output = self.fc[4](self.fc[3](self.fc[2](self.fc[1](self.fc[0](feature)))))
        # output = self.fc(feature)
        # show_tensor(intermediate_output)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()


class Multimodel_with_three(nn.Module):
    def __init__(self, pretrained, num_classes, benign_3, malignant_3, malignant_4a):
        super(Multimodel_with_three, self).__init__()
        self.num_classes = num_classes
        self.model_benign_3 = Multimodel_no_text50_without_class(num_classes=self.num_classes, pretrained=pretrained)
        model_dict = self.model_benign_3.state_dict()
        pretrained_dict = torch.load(benign_3)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_benign_3.load_state_dict(model_dict)
        self.model_malignant_3 = Multimodel_no_text50_without_class(num_classes=self.num_classes, pretrained=pretrained)
        model_dict = self.model_malignant_3.state_dict()
        pretrained_dict = torch.load(malignant_3)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_malignant_3.load_state_dict(model_dict)
        self.model_malignant_4a = Multimodel_no_text50_without_class(num_classes=self.num_classes,
                                                                     pretrained=pretrained)
        model_dict = self.model_malignant_4a.state_dict()
        pretrained_dict = torch.load(malignant_4a)["state_dict"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        self.model_malignant_4a.load_state_dict(model_dict)
        self.fc = nn.Sequential(
            nn.Linear(6144, 2048),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, x_withCDFI):
        x_benign_3 = self.model_benign_3(x, x_withCDFI)
        x_malignant_3 = self.model_malignant_3(x, x_withCDFI)
        x_malignant_4a = self.model_malignant_4a(x, x_withCDFI)
        feature = torch.cat((x_benign_3, x_malignant_3, x_malignant_4a), 1)
        output = self.fc(feature)
        return output

    def Multi_freeze_backbone(self):
        self.model_malignant_3.Multi_freeze_backbone()
        self.model_malignant_4a.Multi_freeze_backbone()
        self.model_benign_3.Multi_freeze_backbone()


class Multimodel_no_text50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        x = self.model(x)[0]
        x = torch.flatten(x, 1)
        intermediate_output = self.fc(x)
        # show_tensor(intermediate_output)
        return intermediate_output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()


class SimCLRStage1(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(SimCLRStage1, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        # projection head
        self.f = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out = self.f(x)
        return out


class SimCLRStage2(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(SimCLRStage2, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        # projection head
        self.f = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128, bias=True)
        )
        # self.f = nn.Sequential(
        #     nn.Linear(2048, 512, bias=False),
        #     # nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 128, bias=True)
        # )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out = self.f(x)
        return out


class SimCLRStage4(nn.Module):
    def __init__(self, model):
        super(SimCLRStage4, self).__init__()
        self.model = model
        # projection head
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


class ShareEncoderModel2(nn.Module):
    def __init__(self, pretrained):
        super(ShareEncoderModel2, self).__init__()
        self.shared_encoder = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        encoded = self.shared_encoder(x)
        return encoded

class ShareEncoderModel2TSNE(nn.Module):
    def __init__(self, pretrained):
        super(ShareEncoderModel2TSNE, self).__init__()
        self.shared_encoder = models.resnet50(pretrained=pretrained)
        # add-8.18
        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048, 128, bias=False),
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128, bias=True)
        )
    def forward(self, x):
        encoded = self.shared_encoder(x)
        out = self.fc1(encoded)
        return out
        # return encoded
class ShareEncoderModel(nn.Module):
    def __init__(self, pretrained):
        super(ShareEncoderModel, self).__init__()
        # self.shared_encoder = models.shufflenet_v2_x1_0(pretrained=True)
        # self.shared_encoder = models.resnet18(pretrained=True)
        self.shared_encoder = models.resnet50(pretrained=pretrained)
        # self.shared_encoder = models.vgg16(pretrained=True)
        # self.shared_encoder = models.densenet121(pretrained=True)
        # self.shared_encoder = models.efficientnet_b0(pretrained=False)

        # # 定义三个独立的全连接层
        # self.fc1 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128)
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128, bias=True)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(512, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(256, 128, bias=True)
        # )

    def forward(self, x):
        encoded = self.shared_encoder(x)
        out = self.fc1(encoded)
        # return encoded
        return out
        # 共享的编码器
        # encoded1 = self.shared_encoder(x1)
        # encoded2 = self.shared_encoder(x2)
        # encoded3 = self.shared_encoder(x3)
        #
        # # 各自的全连接层
        # out1 = self.fc1(encoded1)
        # out2 = self.fc1(encoded2)
        # out3 = self.fc1(encoded3)
        # #
        # return out1, out2, out3
        # return encoded1,encoded2,encoded3

class ShareEncoderModelTSNE(nn.Module):
    def __init__(self, pretrained):
        super(ShareEncoderModelTSNE, self).__init__()
        # self.shared_encoder = models.shufflenet_v2_x1_0(pretrained=True)
        # self.shared_encoder = models.resnet18(pretrained=True)
        self.shared_encoder = models.resnet50(pretrained=pretrained)
        # self.shared_encoder = models.vgg16(pretrained=True)
        # self.shared_encoder = models.densenet121(pretrained=True)
        # self.shared_encoder = models.efficientnet_b0(pretrained=False)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128, bias=True)
        )
        # self.fc1 = nn.Linear(2048, 2)

    def forward(self, x):
        encoded = self.shared_encoder(x)
        out = self.fc1(encoded)
        # return encoded
        return out



class ClassifierModel(nn.Module):
    def __init__(self, model):
        super(ClassifierModel, self).__init__()
        self.model = model
        self.fc = nn.Linear(2048, 2)

        for param in self.model.parameters():
            param.requires_grad = False
        # # ResNet18/ResNet50
        # # for param in self.ENFIEC.layer3.parameters():
        # #     param.requires_grad = True
        # for param in self.ENFIEC.layer4.parameters():
        #     param.requires_grad = True

        # ShuffleNet v2
        # for param in ENFIEC.stage4.parameters():
        #     param.requires_grad = True

        # VGG16
        # for name, param in ENFIEC.named_parameters():
        #     if 'features.24' in name or 'features.26' in name or 'features.28' in name:
        #         param.requires_grad = True
        # for param in ENFIEC.classifier.parameters():
        #     param.requires_grad = True

        # DenseNet121
        # for param in ENFIEC.features.denseblock4.parameters():
        #     param.requires_grad = True

        # EfficientNet b0
        # layers_to_unfreeze = [
        #     'features.7.0.block.0.0.weight',
        #     'features.7.0.block.0.1.weight',
        #     'features.7.0.block.0.1.bias',
        #     'features.7.0.block.1.0.weight',
        #     'features.7.0.block.1.1.weight',
        #     'features.7.0.block.1.1.bias',
        #     'features.7.0.block.2.fc1.weight',
        #     'features.7.0.block.2.fc1.bias',
        #     'features.7.0.block.2.fc2.weight',
        #     'features.7.0.block.2.fc2.bias',
        #     'features.7.0.block.3.0.weight',
        #     'features.7.0.block.3.1.weight',
        #     'features.7.0.block.3.1.bias',
        #     'features.8.0.weight',
        #     'features.8.1.weight',
        #     'features.8.1.bias',
        #     'classifier.1.weight',
        #     'classifier.1.bias',
        #     'fc.weight',
        #     'fc.bias'
        # ]
        # for name, param in ENFIEC.named_parameters():
        #     if name in layers_to_unfreeze:
        #         param.requires_grad = True

    def forward(self, x):
        # x, _ = self.ENFIEC(x)
        x = self.model(x)
        out = self.fc(x)
        return out


class ClassifierBaseModel(nn.Module):
    def __init__(self):
        super(ClassifierBaseModel, self).__init__()
        # self.ENFIEC = models.resnet18(pretrained=True)
        self.model = models.resnet50(pretrained=False)
        # self.ENFIEC = models.resnet50(pretrained=True)
        # self.ENFIEC = models.densenet121(pretrained=True)
        # self.ENFIEC = models.shufflenet_v2_x1_0(pretrained=True)
        # self.ENFIEC = models.vgg16(pretrained=True)
        # self.ENFIEC = models.densenet121(pretrained=True)
        # self.ENFIEC = models.efficientnet_v2_s(pretrained=True)
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(128, 2)
        # )
        self.fc = nn.Linear(2048, 2)

        # ======t_SNE===================
        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048, 128, bias=False),
        # )
        # self.fc2 = nn.Linear(128, 2)
        # ======t_SNE===================

        # self.fc = nn.Linear(1024, 2)
        # self.fc = nn.Linear(1000, 2)
        # self.fc = nn.Linear(512, 2)

    # @autocast(True)
    def forward(self, x):
        x = self.model(x)
        out = self.fc(x)
        # fc1_out = self.fc1(x)
        # out = self.fc2(fc1_out)
        return out

class ClassifierBaseModelTSNE(nn.Module):
    def __init__(self):
        super(ClassifierBaseModelTSNE, self).__init__()
        # self.ENFIEC = models.resnet18(pretrained=True)
        self.model = models.resnet50(pretrained=False)
        # self.fc = nn.Linear(2048, 2)
        # 只能2048到128，不能到512，应该是训练模型就是2048到128
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 128, bias=False),
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(2048, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 128, bias=True)
        # )
        # self.fc2 = nn.Linear(128, 2)
        # self.fc = nn.Linear(1024, 2)
        # self.fc = nn.Linear(1000, 2)
        # self.fc = nn.Linear(512, 2)

    # @autocast(True)
    def forward(self, x):
        x = self.model(x)
        fc1_out = self.fc1(x)
        return fc1_out
        # return x

class Multimodel_Multiexperts(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_Multiexperts, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.ca1 = ChannelAttention(512)
        self.ca2 = ChannelAttention(1024)
        self.ca3 = ChannelAttention(2048)
        self.ca4 = ChannelAttention(4096)
        self.proj1 = nn.Linear(512, 4096)
        self.proj2 = nn.Linear(1024, 4096)
        self.proj3 = nn.Linear(2048, 4096)
        self.proj4 = nn.Linear(4096, 4096)
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x, x_withCDFI):
        x, x_1, x_2, x_3, x_4 = self.model(x)
        y, y_1, y_2, y_3, y_4 = self.model_CDFI(x_withCDFI)
        x1_y1 = torch.cat((x_1, y_1), 1)
        x1_y1 = self.ca1(x1_y1)
        x1_y1 = self.avg_pooling(x1_y1)
        x1_y1 = torch.flatten(x1_y1, 1)
        x1_y1 = self.proj1(x1_y1)
        x2_y2 = torch.cat((x_2, y_2), 1)
        x2_y2 = self.ca2(x2_y2)
        x2_y2 = self.avg_pooling(x2_y2)
        x2_y2 = torch.flatten(x2_y2, 1)
        x2_y2 = self.proj2(x2_y2)
        x3_y3 = torch.cat((x_3, y_3), 1)
        x3_y3 = self.ca3(x3_y3)
        x3_y3 = self.avg_pooling(x3_y3)
        x3_y3 = torch.flatten(x3_y3, 1)
        x3_y3 = self.proj3(x3_y3)
        x4_y4 = torch.cat((x_4, y_4), 1)
        x4_y4 = self.ca4(x4_y4)
        x4_y4 = self.avg_pooling(x4_y4)
        x4_y4 = torch.flatten(x4_y4, 1)
        x4_y4 = self.proj4(x4_y4)
        output = x1_y1 + x2_y2 + x3_y3 + x4_y4
        output = self.fc(output)
        #x = torch.flatten(x, 1)
        #y,y_1,y_2,y_3,y_4 = self.model_CDFI(x_withCDFI)
        #x_withCDFI = torch.flatten(x_withCDFI, 1)
        #feature = torch.cat((x, x_withCDFI), 1)
        return output


class Multimodel_no_text50_2(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50_2, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),  # 4096+2    #4096+59
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 256),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes))

    def forward(self, x, x_withCDFI):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x_withCDFI = self.model_CDFI(x_withCDFI)
        x_withCDFI = torch.flatten(x_withCDFI, 1)
        feature = torch.cat((x, x_withCDFI), 1)
        output1 = self.fc1(x)
        output = self.fc2(output1)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()

    def Multi_Unfreeze_backbone(self):
        self.model.Unfreeze_backbone()
        self.model_CDFI.Unfreeze_backbone()
        for param in self.fc1:
            param.requires_grad = True


class Multimodel_no_text50_3(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text50_3, self).__init__()
        self.num_classes = num_classes
        self.model = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.model_CDFI = resnet50(num_classes=self.num_classes, pretrained=pretrained)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),  # 4096+2    #4096+59
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 256),  # 4096+2    #4096+59
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes))

    def forward(self, x, x_withCDFI):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x_withCDFI = self.model_CDFI(x_withCDFI)
        x_withCDFI = torch.flatten(x_withCDFI, 1)
        feature = torch.cat((x, x_withCDFI), 1)
        output1 = self.fc1(x_withCDFI)
        output = self.fc2(output1)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDFI.freeze_backbone()

    def Multi_Unfreeze_backbone(self):
        self.model.Unfreeze_backbone()
        self.model_CDFI.Unfreeze_backbone()
        for param in self.fc1:
            param.requires_grad = True


class Multimodel_no_text18(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(Multimodel_no_text18, self).__init__()
        self.num_classes = num_classes
        self.model = resnet18(num_classes=num_classes, pretrained=pretrained)
        self.model_CDIF = resnet18(num_classes=num_classes, pretrained=pretrained)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # 4096+92    #4096+59
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x, x_withCDFI):
        x = self.model(x)
        x = torch.flatten(x, 1)
        #x_withCDFI = self.model_CDIF(x_withCDFI)
        #x_withCDFI = torch.flatten(x_withCDFI, 1)
        #output = torch.cat((x, x_withCDFI), 1)
        output = self.fc(x)
        return output

    def Multi_freeze_backbone(self):
        self.model.freeze_backbone()
        self.model_CDIF.freeze_backbone()

    def Multi_Unfreeze_backbone(self):
        self.model.Unfreeze_backbone()
        self.model_CDIF.Unfreeze_backbone()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc(avg_pool.view(avg_pool.size(0), -1)).view(avg_pool.size(0), -1, 1, 1)
        max_out = self.fc(max_pool.view(max_pool.size(0), -1)).view(max_pool.size(0), -1, 1, 1)
        attention = self.sigmoid(avg_out + max_out)
        return attention * x


class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)

        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


if __name__ == "__main__":
    multimodel = Multimodel_Multiexperts(pretrained=True, num_classes=2)
    input_image = torch.randn(4, 3, 256, 256)
    input_image2 = torch.randn(4, 3, 256, 256)
    x, x1, x2, x3, x4 = multimodel(input_image, input_image2)
    print(x.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
