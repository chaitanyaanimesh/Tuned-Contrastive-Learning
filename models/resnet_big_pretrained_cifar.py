"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self,modelName, block, num_blocks,num_classes,dim_in,pretrained, in_channel=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.feat_dim=128  #256 for SimCLR and 128 for SupCon
        self.bn1=None
        self.layer1=None
        self.layer3=None
        self.layer4=None
        self.avgpool=None
        self.linear=None
        self.head=None
        self.model= None
        
        
        if(pretrained):
            self.model = self.loadPretrainedModel(modelName)
            
            self.conv1 = self.model.conv1
            self.bn1 = self.model.bn1
            self.layer1 = self.model.layer1
            self.layer2 = self.model.layer2
            self.layer3 = self.model.layer3
            self.layer4 = self.model.layer4
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(dim_in, num_classes)
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in),nn.ReLU(inplace=True),nn.Linear(dim_in, self.feat_dim))
        else:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,   #Kernel Size=7. Making it 3 for cifar
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(dim_in, num_classes)
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in),nn.ReLU(inplace=True),nn.Linear(dim_in, self.feat_dim))

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves
            # like an identity. This improves the model by 0.2~0.3% according to:
            # https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)
    
    def loadPretrainedModel(self,modelName):
        if(modelName=='resnet18'):
            return models.resnet18(pretrained=True)
        elif(modelName=='resnet34'):
            return models.resnet34(pretrained=True)
        elif(modelName=='resnet50'):
            return models.resnet50(pretrained=True)
        elif(modelName=='resnet101'):
            return models.resnet101(pretrained=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


   
    #Added
    def freeze_projection(self):
        self.conv1.requires_grad_(False)
        self.bn1.requires_grad_(False)
        self.layer1.requires_grad_(False)
        self.layer2.requires_grad_(False)
        self.layer3.requires_grad_(False)
        self.layer4.requires_grad_(False)

    def _forward_impl_encoder(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out

    def forward_constrative(self, x):
        # Implement from the encoder E to the projection network P
        x = self._forward_impl_encoder(x)

        x = self.head(x)
        # Normalize to unit hypersphere
        x = F.normalize(x, dim=1)

        return x

    def forward(self, x):
        # Implement from the encoder to the decoder network
        x = self._forward_impl_encoder(x)
        return self.linear(x)


def resnet18(num_classes,dim_in,pretrained,**kwargs):
    return ResNet('resnet18',BasicBlock, [2, 2, 2, 2],num_classes,dim_in,pretrained, **kwargs)


def resnet34(num_classes,dim_in,pretrained,**kwargs):
    return ResNet('resnet34',BasicBlock, [3, 4, 6, 3],num_classes,dim_in,pretrained, **kwargs)


def resnet50(num_classes,dim_in,pretrained,**kwargs):
    return ResNet('resnet50',Bottleneck, [3, 4, 6, 3],num_classes,dim_in,pretrained, **kwargs)


def resnet101(num_classes,dim_in,pretrained,**kwargs):
    return ResNet('resnet101',Bottleneck, [3, 4, 23, 3],num_classes,dim_in,pretrained, **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet20': [resnet18,512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet56': [resnet50,2048],
    'resnet101': [resnet101, 2048],
}


    
def get_resnet_contrastive_big_pretrained_cifar(name='resnet18', num_classes=10, pretrained=False):


    if name in model_dict:
        return model_dict[name][0](num_classes,model_dict[name][1],pretrained)
    else:
        raise ValueError('Model name %s not found'.format(name))