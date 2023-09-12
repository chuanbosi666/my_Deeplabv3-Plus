import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):  # 通过这一步来获取Deeplabv3模型，可以获得不同的输出，也就是下面的类的feature的不同值
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):  #构建完整的Deeplabv3+模块 ，从上面的__all__ = ["DeepLabV3"]可以得到，DeepLabHeadV3Plus继承了DeepLabHeadV3
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):  #设定输入通道，类别数，ASPP的扩张率
        #这里说明一下，可从论文看到，使用的DCNN模块是Deeplabv3，'Low_level'对应的是前四层的输出
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),  # 对project的定义，一个卷积BN激活
        )

        self.aspp = ASPP(in_channels, aspp_dilate)  # ASPP模块的引入

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )  #分类器
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )  # 将特征字典中的'low_level'内容传进去
        output_feature = self.aspp(feature['out']) # 将特征的out部分传进去
        #将上面的部分继续双线性插值，之后再将低维度特征处理后的和较高维度处理后的进行拼接，在第一维度进行处理，经过分类器处理
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):  # 构建head部分
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(  # 构建一个分类器
            ASPP(in_channels, aspp_dilate),  #首先是ASPP模块
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  #进行3x3卷积，填充为一，不是设置偏置，保证输入和输出不变，提取更高级别的特征表示。
            nn.BatchNorm2d(256),  #规范输出，以此加速模型训练并提高模型的稳定性。
            nn.ReLU(inplace=True),  #引入非线性变换并增加模型的表达能力
            nn.Conv2d(256, num_classes, 1)  # 1x1卷积分类，就是将256通道数转变成的我们的类别数
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )  # 将特征的字典中'out'的内容传入该函数中

    def _init_weight(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):  #构建扩张可分离卷积
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__() # 放入父类
        self.body = nn.Sequential(  # 构建我们的深度可分离网络的主干部分，就是先进行深度可分离卷积，再进行PointWise卷积
            # Separable Conv  和正常卷积的参数一样，就是加上了扩张率和对group的处理指定
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )  #输入通道，输出通道，卷积核大小为1，步幅为1，不进行填充，设置偏置
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):  #初始化权重
        for m in self.modules():  # 遍历获取模块
            if isinstance(m, nn.Conv2d):  # 对卷积的权重使用kaiming归一化
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  # 对BN层和群归一化进行处理，初始化为1和0
                nn.init.constant_(m.weight, 1)  # 此外，nn.GroupNorm是将通道集结成群进行归一化处理
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):  #设定ASPP模块的卷积部分
    def __init__(self, in_channels, out_channels, dilation):
        modules = [  # 输入输出通道，卷积核为3x3大小，填充大小为扩张率，不设置偏置
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):  #设置ASPP的池化部分
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  #进行自适应平均池化，输出大小为1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1的卷积
            nn.BatchNorm2d(out_channels), #BN层
            nn.ReLU(inplace=True)) #激活函数

    def forward(self, x):
        size = x.shape[-2:]  #取出图片的大小
        x = super(ASPPPooling, self).forward(x) # 对其进行自适应平均池化操作等
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False) # 再进行双线性插值还原图片

class ASPP(nn.Module):  #构建完整的ASPP模块
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256  #设置输出通道
        modules = []  #构建一个列表
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))  #传进去1x1卷积BN激活函数

        rate1, rate2, rate3 = tuple(atrous_rates)  #获取我们的扩张率：12、24、36
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        # 这四行代码是构建我们的ASPP模块，分别是扩张率为12、24、36的扩张卷积，还有一个图片的池化操作，就此我们的ASPP模块构建完成
        #一个1x1卷积模块，扩张率分别为12、24、36的扩张卷积，还有一个对图片进行平均池化的部分
        self.convs = nn.ModuleList(modules)  # 封入其中

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), # 输入通道数为1280，输出通道数为256，使用1x1卷积
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)  # 进行dropout

    def forward(self, x):
        res = []  # 创建空列表
        for conv in self.convs: # 将上面的构建的模块进行遍历
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # 在第一个维度上进行拼接
        return self.project(res)  # 将拼接后的在经过处理



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:  #如果 module 是一个卷积层并且卷积核大小大于1
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias) # 将卷积层替换为 Atrous Separable Convolution 层
    for name, child in module.named_children():  # 递归调用该函数，逐渐替换为Atrous Separable Convolution 层
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module