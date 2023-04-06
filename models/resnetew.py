import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Conv2dExpWeighted(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, exp_weight=False, t=1.0):
        self.exp_weight = exp_weight
        self.t = t
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        if self.exp_weight:
            w = torch.exp(torch.abs(self.weight) * self.t)
            
            w = w/ torch.max(w)
            w *= self.weight
            self.exp_w = nn.Parameter(w)
            out = nn.functional.conv2d(x, self.exp_w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            out = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out



def conv3x3ew(in_planes, out_planes, stride=1, exp_weight=False, t=1.0):
    return Conv2dExpWeighted(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, exp_weight=exp_weight, t=t)


class PreActBlockew(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, exp_weight=False, t=1.0):
        super(PreActBlockew, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3ew(in_planes, planes, stride, exp_weight=exp_weight, t=t)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3ew(planes, planes, exp_weight=exp_weight, t=t)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2dExpWeighted(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False, exp_weight=exp_weight, t=t)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


inps, outs = [], []#加的，考虑减掉
def layer_hook(module, inp, out):
    inps.append(inp[0].data.cpu().np())
    outs.append(out.data.cpu().np())



class ResNetew(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,exp_weight=False,t=1.0):
        super(ResNetew, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3ew(3, 64,1,exp_weight=exp_weight,t=t)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,exp_weight=exp_weight,t=t)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,exp_weight=exp_weight,t=t)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,exp_weight=exp_weight,t=t)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,exp_weight=exp_weight,t=t)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,exp_weight,t):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,exp_weight,t))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def freeze_hidden_layers(self):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.bn1)
        self._freeze_layer(self.layer1)
        self._freeze_layer(self.layer2)
        self._freeze_layer(self.layer3)
        self._freeze_layer(self.layer4)

    def unfreeze_model(self):
        self._freeze_layer(self.conv1, freeze=False)
        self._freeze_layer(self.bn1, freeze=False)
        self._freeze_layer(self.layer1, freeze=False)
        self._freeze_layer(self.layer2, freeze=False)
        self._freeze_layer(self.layer3, freeze=False)
        self._freeze_layer(self.layer4, freeze=False)
        self._freeze_layer(self.linear, freeze=False)

    def embed_in_n_layer(self, n):
        self._freeze_layer(self.conv1)
        self._freeze_layer(self.bn1)
        if n == 1:
            self._freeze_layer(self.layer1)
        elif n == 2:
            self._freeze_layer(self.layer2)
        elif n == 3:
            self._freeze_layer(self.layer3)
        elif n == 4:
            self._freeze_layer(self.layer4)
        else:
            self._freeze_layer(self.linear)

    def _freeze_layer(self, layer, freeze=True):
        if freeze:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18ew(num_classes=10,exp_weight=False, t=1.0):
    return ResNetew(PreActBlockew, [2, 2, 2, 2])


def get_params_to_ew(arch, net):#待修改
    if arch == "resnet18":
        return [
            (net.module.conv1, 'weight'),
            (net.module.bn1, 'weight'),

            (net.module.layer1[0].conv1, 'weight'),
            (net.module.layer1[0].bn1, 'weight'),
            (net.module.layer1[0].conv2, 'weight'),
            (net.module.layer1[0].bn2, 'weight'),
            
            #add later
            (net.module.layer1[1].conv1, 'weight'),
            (net.module.layer1[1].bn1, 'weight'),
            (net.module.layer1[1].conv2, 'weight'),
            (net.module.layer1[1].bn2, 'weight'),

            (net.module.layer2[0].conv1, 'weight'),
            (net.module.layer2[0].bn1, 'weight'),
            (net.module.layer2[0].conv2, 'weight'),
            (net.module.layer2[0].bn2, 'weight'),

            (net.module.layer2[0].shortcut[0], 'weight'),

            (net.module.layer2[1].conv1, 'weight'),
            (net.module.layer2[1].bn1, 'weight'),
            (net.module.layer2[1].conv2, 'weight'),
            (net.module.layer2[1].bn2, 'weight'),

            (net.module.layer3[0].conv1, 'weight'),
            (net.module.layer3[0].bn1, 'weight'),
            (net.module.layer3[0].conv2, 'weight'),
            (net.module.layer3[0].bn2, 'weight'),

            (net.module.layer3[0].shortcut[0], 'weight'),

            (net.module.layer3[1].conv1, 'weight'),
            (net.module.layer3[1].bn1, 'weight'),
            (net.module.layer3[1].conv2, 'weight'),
            (net.module.layer3[1].bn2, 'weight'),

            (net.module.layer4[0].conv1, 'weight'),
            (net.module.layer4[0].bn1, 'weight'),
            (net.module.layer4[0].conv2, 'weight'),
            (net.module.layer4[0].bn2, 'weight'),

            (net.module.layer4[0].shortcut[0], 'weight'),

            (net.module.layer4[1].conv1, 'weight'),
            (net.module.layer4[1].bn1, 'weight'),
            (net.module.layer4[1].conv2, 'weight'),
            (net.module.layer4[1].bn2, 'weight')

        ]


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())

# test()
print("resnetew")