import torch.nn as nn
from .ConvNext import make_convnext_model

class two_view_net(nn.Module):
    def __init__(self, class_num, block=4, M=32 ,return_f=False, resnet=False):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(num_class=class_num, block=block,M=M, return_f=return_f, resnet=resnet)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_1(x2)
        return y1, y2


class three_view_net(nn.Module):
    def __init__(self, class_num, share_weight = False,block=4,M=32 ,return_f=False, resnet=False):
        super(three_view_net, self).__init__()
        self.share_weight = share_weight
        self.model_1 = make_convnext_model(num_class=class_num, block=block,M=M, return_f=return_f, resnet=resnet)


        if self.share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = make_convnext_model(num_class=class_num,  block=block,M=M, return_f=return_f, resnet=resnet)



    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)

        if x3 is None:
            y3 = None
        else:
            y3 = self.model_1(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = self.model_2(x4)
        return y1, y2, y3, y4


def make_model(opt):
    if opt.views == 2:
        model = two_view_net(opt.nclasses, block=opt.block,M=opt.M,return_f=opt.triplet_loss,resnet=opt.resnet)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, share_weight=opt.share,block=opt.block,M=opt.M,return_f=opt.triplet_loss, resnet=opt.resnet)
    return model
