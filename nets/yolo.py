from collections import OrderedDict

import torch
import torch.nn as nn

from nets.CSPdarknet import darknet53


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    '''
    conv+bn+LeakRelu 模块
    :param filter_in: 输入channel
    :param filter_out: 输出channel
    :param kernel_size: kersize大小
    :param stride: 跨距默认1
    :return:
    '''
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # kersize=pool_size
        # stride=1
        # pool=pool_size//2
        # 这么计算下来,对于13x13的特征图,每个pool_sizes的输出都是13x13的特征图...
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        # 输出先是pool=13的特征图结果,然后pool=9,pool=5,pool=1,堆叠
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        卷积+上采样, 输出特征图恢复成2倍长宽大小
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        '''
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    '''
    三层卷积,分别为1x1压缩,3x3特征提取,1x1还原
    三次卷积中每个都是conv_bn_relu
    :param filters_list: 两个元素组成的list filters_list[0]表示输出n,filters_list[1]表示中间层深度
    :param in_filters: 输入深度
    :return:
    '''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    '''
    1x1降维 3x3升维 1x1降维 3x3升维 1x1降维为输出
    :param filters_list: [0]:输出channel [1]:中间层升维的深度
    :param in_filters: 输入channel长度
    :return:
    '''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    '''
    3x3卷积加深通道+1x1卷积输出 75类别
    :param filters_list:两个元素组成的list filters_list[0]表示中间层深度,filters_list[1]表示输出
    :param in_filters:输入深度
    :return:
    '''
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成CSPdarknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone   = darknet53(pretrained)

        self.conv1      = make_three_conv([512,1024],1024)
        self.SPP        = SpatialPyramidPooling()
        # 512深度直接翻了4倍变成2048
        # 然而spp后特征图大小还是固定的13x13
        # 注意,因为这和标准的spp网络不同
        # 后面没接全连接,只是特征图,如果是标准的spp或者fasterrcnn
        # 则spp后输出的不同特征图尺寸则应该直接展平了
        self.conv2      = make_three_conv([512,1024],2048)

        self.upsample1          = Upsample(512,256)
        self.conv_for_P4        = conv2d(512,256,1)
        self.make_five_conv1    = make_five_conv([256, 512],512)

        self.upsample2          = Upsample(256,128)
        self.conv_for_P3        = conv2d(256,128,1)
        self.make_five_conv2    = make_five_conv([128, 256],256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3         = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)],128)

        self.down_sample1       = conv2d(128,256,3,stride=2)
        self.make_five_conv3    = make_five_conv([256, 512],512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2         = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)],256)

        self.down_sample2       = conv2d(256,512,3,stride=2)
        self.make_five_conv4    = make_five_conv([512, 1024],1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1         = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)],512)


    def forward(self, x):
        #  backbone
        #   获得三个有效特征层，他们的shape分别是：
        #   x2:52,52,256
        #   x1:26,26,512
        #   x0:13,13,1024
        x2, x1, x0 = self.backbone(x)

        # conv1,三次卷积,输入1024,输出512
        P5 = self.conv1(x0)
        # SPP多尺度maxpool,特征图尺度不变,深度变为4倍,变成[13,13,2048]
        P5 = self.SPP(P5)
        # [13,13,2048]->[13,13,512]
        P5 = self.conv2(P5)
        # [13,13,512]->[26,26,256]
        P5_upsample = self.upsample1(P5)

        # [26,26,512]->[26,26,256]
        P4 = self.conv_for_P4(x1)
        # 融合成[26,26,512]
        P4 = torch.cat([P4,P5_upsample],axis=1)
        # [26,26,512]->[26,26,256]
        P4 = self.make_five_conv1(P4)

        # [26,26,256]->[52,52,128]
        P4_upsample = self.upsample2(P4)
        # [52,52,256]->[52,52,128]
        P3 = self.conv_for_P3(x2)
        # [52,52,256]
        P3 = torch.cat([P3,P4_upsample],axis=1)
        # [52,52,256]->[52,52,128]
        P3 = self.make_five_conv2(P3)

        # [52,52,128]->[26,26,256]
        P3_downsample = self.down_sample1(P3)
        # [26,26,512]
        P4 = torch.cat([P3_downsample,P4],axis=1)
        # [26,26,512]->[26,26,256]
        P4 = self.make_five_conv3(P4)

        # [26,26,256]->[13,13,512]
        P4_downsample = self.down_sample2(P4)
        # [13,13,1024]
        P5 = torch.cat([P4_downsample,P5],axis=1)
        # [13,13,1024]->[13,13,512]
        P5 = self.make_five_conv4(P5)

        #---------------------------------------------------#
        #   第三个特征层 P3:[52,52,128]
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层 P4:[26,26,256]
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层 P5:[13,13,512]
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2

