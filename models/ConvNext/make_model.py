import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter


class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class BasicConv_For_ADIB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv_For_ADIB, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return x


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()

    def forward(self, x):
        std = x.std(dim=1, keepdim=True)
        return std


class ADPool(nn.Module):
    def __init__(self):
        super(ADPool, self).__init__()
        self.std_pool = StdPool()
        self.weight = nn.Parameter(torch.rand(2))

    def forward(self, x):
        std_pool = self.std_pool(x)
        # max_pool = torch.max(x,1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        weight = torch.sigmoid(self.weight)
        out = 1 / 2 * (std_pool + avg_pool) + weight[0] * std_pool + weight[1] * avg_pool
        return out


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ADPool()
        self.conv = BasicConv_For_ADIB(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class ADIB_block(nn.Module):
    def __init__(self):
        super(ADIB_block, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


class Attentions(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Attentions, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B,
                                                                                                          -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-6)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
            torch.abs(counterfactual_feature) + 1e-6)

        counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature


class CIB_block(nn.Module):
    def __init__(self, in_planes, block=4, M=32):
        super(CIB_block, self).__init__()
        self.in_planes = in_planes
        self.block = block
        self.M = M
        self.bap = BAP(pool='GAP')
        self.attentions = Attentions(self.in_planes, self.M, kernel_size=1)

    def forward(self, x):
        part = {}
        part_attention_maps = {}
        part_normal_feature = {}
        part_counterfactual_feature = {}
        for i in range(self.block):
            # part[i] = x[:, :, i].view(x.size(0), -1)
            part[i] = x[:, :, :, :, i]
            part_attention_maps[i] = self.attentions(part[i])  # attention_maps[8,32,8,8]
            part_normal_feature[i], part_counterfactual_feature[i] = self.bap(part[i], part_attention_maps[i])
        return part_normal_feature, part_counterfactual_feature


class build_CCR(nn.Module):
    def __init__(self, num_classes, block=4, M=32, return_f=False, resnet=False):
        super(build_CCR, self).__init__()
        self.return_f = return_f
        if resnet:
            resnet_name = "resnet50"
            print('using model_type: {} as a backbone'.format(resnet_name))
            self.in_planes = 2048
            self.backbone = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.backbone = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.block = block
        self.M = M
        self.ADIB_layer = ADIB_block()
        self.CIB_layer = CIB_block(self.in_planes, self.block, self.M)
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        for i in range(self.block * 2):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes * self.M, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        gap_feature, part_features = self.backbone(x)
        ADIB_features = self.ADIB_layer(part_features)
        convnext_feature = self.classifier1(gap_feature)
        ADIB_list = []
        for i in range(self.block):
            ADIB_list.append(ADIB_features[i])
        ADIB_attention_features = torch.stack(ADIB_list, dim=4)
        nfeature, cfeature = self.CIB_layer(ADIB_attention_features)

        if self.block == 0:
            y = []
        elif self.training:
            y, y_counterfactual = self.part_classifier(self.block, nfeature, cfeature, cls_name='classifier_mcb')
        else:
            y = self.part_classifier(self.block, nfeature, cfeature, cls_name='classifier_mcb')

        if self.training:
            y = y + [convnext_feature]
            y = y + y_counterfactual
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            y = torch.cat([y, ffeature], dim=2)
        return y

    def part_classifier(self, block, nf, cf, cls_name='classifier_mcb'):
        predict_normal = {}
        predict_counterfactual = {}
        for i in range(block):
            name_normal = cls_name + str(i + 1)
            c_normal = getattr(self, name_normal)
            predict_normal[i] = c_normal(nf[i])

            name_counterfactual = cls_name + str(i + 1 + block)
            c_name_counterfactual = getattr(self, name_counterfactual)
            counterfactual_classifier = c_name_counterfactual(cf[i])
            predict_counterfactual[i] = (
            predict_normal[i][0] - counterfactual_classifier[0], predict_normal[i][1] - counterfactual_classifier[1])
        y_normal = []
        y_counterfactual = []
        for i in range(block):
            y_normal.append(predict_normal[i])
            y_counterfactual.append(predict_counterfactual[i])
        if not self.training:
            return torch.stack(y_normal, dim=2)
        return y_normal, y_counterfactual


def make_CCR_model(num_class, block=4, M=32, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_CCR(num_class, block=block, M=M, return_f=return_f, resnet=resnet)
    return model
