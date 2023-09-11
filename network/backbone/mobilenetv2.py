from torch import nn
try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}  # 模型的预训练权重


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    确保每一层都会被8整除
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:因子
    :param min_value:最小值
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 新的value值是从最小值，value加上二分之一的因子再整除于因子再乘因子之中取最大值
    # Make sure that round down does not go down by more than 10%.
    #确保四舍五入的幅度不超过10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):  # 从字面可知这个的函数有什么的组成的
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        #padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def fixed_padding(kernel_size, dilation):  # 修正padding的大小
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1) # 首先计算有效卷积核的大小
    pad_total = kernel_size_effective - 1  # 表示了在进行空洞卷积时的填充数
    pad_beg = pad_total // 2  # 输入的两侧需要添加的修正大小
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)   #表示在输入的上、下、左和右四个方向上需要添加的修正大小。
# 对应于下面的类，就是MobileNetV2的基本模块，
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # 进行一个判断

        hidden_dim = int(round(inp * expand_ratio))  # 输入通道深度乘扩展率，从下面的代码可以看出，扩张率分别为1、6、6、6、6、6、6
        self.use_res_connect = self.stride == 1 and inp == oup  # 从下面的文件可知，输入通道等于输出通道，并且第一五七层时会use_res_connect等于1，两者都会满足

        layers = []
        if expand_ratio != 1:  #通过上面的代码可以看出我们模型的第一层是仅有卷积BNrelu，而后面的五层是在原来的基础上加上了卷积BN
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))  # 扩张卷积

        layers.extend([
            # dw 深度可分离卷积
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear  线性投影卷积
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),# 输入通道数为hidden_dim，输出为oup，kernel大小为1，步幅为1，填充为0
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)  #，上面的扩张卷积，深度可分离卷积，线性投影卷积构成了我们的InvertedResidual模块

        self.input_padding = fixed_padding( 3, dilation )  # 在本文件中的扩张率为1，不进行空洞卷积，也就是填充为1

    def forward(self, x):
        x_pad = F.pad(x, self.input_padding) # 先对图像进行填充四边各1个0
        if self.use_res_connect:  # 在157层会使用残差模块，在其他层会仅经过上面的构建的架构
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280  # 设定的最后一层的输出
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t（Expansion Factor）
                # c（Output Channels）
                # n（Number of Blocks）
                # s（Stride）
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        # 调用这个函数会让我们的输入通道保证可以被round_nearest，也就是8正除
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        #调用这个函数和上面是相同的道理
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]  # 先设定好我们的特征处理
        current_stride *= 2  # 会将步幅乘于二
        dilation=1
        previous_dilation = 1

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:  # 对其进行遍历，遍历七遍
            output_channel = _make_divisible(c * width_mult, round_nearest) # c是输出通道，进行处理，保证的可以被8整除
            previous_dilation = dilation  # 为1
            if current_stride == output_stride: # 会在第1、5、7层处相等
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)  # 再对channel乘于1.0

            for i in range(n):  # n会是1、2、3、4、3、3、1的分布
                if i==0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:  # 会构建InvertedResidual，传进去输入通道，输出通道，步幅设定为1，扩张率为1，膨胀率为t，t的值为1、6、6、6、6、6、6
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel
        # building last several layers  构建最后层
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)  # 这个星号的含义是将features拆出来给重新罗列出来

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        ) #定义分类器

        # weight initialization 权重初始化
        for m in self.modules():  #self.modules不用自己定义，会直接对当前文件进行遍历，获取每一个模块，包括当前模块
            if isinstance(m, nn.Conv2d):  # 对实例进行判断，是否2D卷积
                # 这是对卷积层的权重进行kaiming归一化，'fan_out' 模式：标准差为 1 / sqrt(fan_out)，其中 fan_out 是输出通道数。
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:  # 存在偏置的话，会将偏置初始化
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):  #存在BN，会对权重和偏置进行初始化
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):  # 存在线性层的话，
                nn.init.normal_(m.weight, 0, 0.01)  # 将其归一化成正态分布的随机值，均值为0，随机差为0.01
                nn.init.zeros_(m.bias) # 偏置归零

    def forward(self, x):  #前向传播构建模型
        x = self.features(x)
        x = x.mean([2, 3])  # 在第2、3维度求平均
        x = self.classifier(x)  # 再经过一个分类器
        return x  # 完成构建


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)  # 调用上面的类，传进去参数
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)  #是否导入预训练权重
    return model
