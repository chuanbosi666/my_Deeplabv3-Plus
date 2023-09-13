from tqdm import tqdm
# 通过这个导入，将我们的模型变成一个可调用的模型，这样就可以直接调用模型进行训练
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
# 导入数据集，分别是VOC和Cityscapes数据集
from datasets import VOCSegmentation, Cityscapes
# 导入数据增强模块，分别是Resize、RandomScale、RandomCrop、RandomHorizontalFlip、ToTensor、Normalize，并且将其重命名为et
from utils import ext_transforms as et
# 导入评价指标，是StreamSegMetrics
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
#从utils中导入可视化模块Visualizer
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    # 在解码器和ASPP部分应用可分离卷积，默认为False
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts): # 设定数据集
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':  # 当使用VOC数据集时，使用VOC数据集的数据增强
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),  # 缩放倍数在0.5到2之间
            # 设定的裁剪尺寸，另一个参数是在图片的尺寸在小于裁剪尺寸时，并且参数设置为true时，会进行填充
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),  # 进行水平翻转
            et.ExtToTensor(),  #转变成tensor
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),  #设置归一化的均值和方差，均值的三个值代表滑动均值，批次均值，实例均值
        ])
        if opts.crop_val:  #这是对验证集的数据增强
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,  #这是对训练集的设定，传进去的参数是数据集的根目录，年份，数据集的类型，是否下载，数据增强
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,  # 同上
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':# 当使用Cityscapes数据集时，使用Cityscapes数据集的数据增强，和上面有些许不同，主要是因为Cityscapes数据集的数据增强是固定的
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        # 这是对验证集的数据增强
        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        # 这是对训练集的设定，传进去的参数是数据集的根目录，年份，数据集的类型，是否下载，数据增强，下面的验证集也是如此
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst

# 接下来就是对模型进行验证
def validate(opts, model, loader, device, metrics, ret_samples_ids=None): # 传进去的参数是opts，模型，数据集，设备，评价指标，返回的样本id
    """Do validation and return specified samples"""
    metrics.reset()  # 对评价指标进行重置
    ret_samples = []  #创建一个列表，用于存储返回的样本
    if opts.save_val_results:  # 如果保存验证结果为真，就创建一个文件夹
        if not os.path.exists('results'):
            os.mkdir('results')  # 创建文件夹
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])  # 对图片进行反归一化，是为了将图片转换成PIL格式
        img_id = 0

    with torch.no_grad():   # 不进行梯度计算
        for i, (images, labels) in tqdm(enumerate(loader)):  # tqdm是一个进度条，用于显示进度,将数据集进行遍历

            images = images.to(device, dtype=torch.float32) # 将图片和标签转换成tensor
            labels = labels.to(device, dtype=torch.long)  # 将标签转换成long类型

            outputs = model(images)  # 这里传入的模型就是deeplabv3,使用的是mobilenet作为骨干网络
            # 这是模型输出后的处理，先分离将张量创建并分离出来，在第一维度上寻找最大值的张量及其对应的索引的张量，放置到CPU上去，再转换为numpy
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()  # 将标签放置到CPU上去，再转换为numpy

            metrics.update(targets, preds)  # 将我们的target和预测传进去，更新评价指标
            # get vis samples 如果返回的样本id不为空，并且i在返回的样本id中，就将图片，标签，预测结果放入列表中
            if ret_samples_ids is not None and i in ret_samples_ids:  
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:  # 对于保存验证结果为真的情况
                for i in range(len(images)):  # 将图片，标签，预测结果保存下来
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
                    # 将图片转换成PIL格式，再转换成numpy，再转换成uint8，target和pred也是如此
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    # 保存图片，标签，预测结果，叠加结果，% img_id是为了将图片的id保存下来
                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():  # 这部分是主函数
    opts = get_argparser().parse_args()  # 将参数部分变成一个列表，称为opts
    if opts.dataset.lower() == 'voc':  # 这是对数据集的设定，当数据集为voc时，类别数为21；当数据集为cityscapes时，类别数为19
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization 设置可视化部分
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id  # 设置GPU的ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
    print("Device: %s" % device)

    # Setup random seed 设置随机种子
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader  设置数据加载器
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1
    # 这里的train_dst和val_dst是数据集，train_loader和val_loader是数据加载器
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(  # 这里的DataLoader是pytorch中的数据加载器，用于将数据集加载到模型中
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(  # 设置验证的dataloader
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling) 设置模型
    # 这里的model是deeplabv3plus_mobilenet，是通过导入network文件夹，使用的是mobilenet作为骨干网络
    # 传进来的参数为deeplabv3plus_mobilenet，类别数为默认值，输出步长为16
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)  
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)  # 但参数使用分离卷积并且model中含有plus会将分类器的卷积替换成深度可分离卷积
    utils.set_bn_momentum(model.backbone, momentum=0.01) # 设置BN的动量

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer 设置优化器
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':  #根据不同的策略设定学习率策略，传入相应的权重值
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion  设定不同的损失函数
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model  保存模型
        """
        torch.save({  # 会存储下面的东西，当前迭代数，模型，优化器，调度器以及最好的得分信息
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')  # 创建文件，将checkpoints保存下来
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):  # 对条件进行判断，ckpt是true并且上面创建文件成功
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))  # 加载文件
        model.load_state_dict(checkpoint["model_state"])# 将文件里的model_state这一部分加载出来
        model = nn.DataParallel(model)  # 将模型进行并行训练
        model.to(device)  
        if opts.continue_training:  # 如果在现有的模型继续训练的话，会加载优化器，学习率调度器，当前的迭代次数，最好的得分
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)  # 打印信息
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory  没有的话，就会释放内存
    else:
        print("[!] Retrain")  # 进行再训练
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    # 生成随机整数的数组，范围是0到验证数据集加载器的长度，大小为vis_num_samples，数据类型为np的32整型
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:  # 对测试进行设置
        model.eval()
        # 将我们的参数传入到验证中，从而获得我们的评价指标得分和返回的样本
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))  # 打印下来验证的评价指标得分
        return

    interval_loss = 0  # 设置间隔损失为0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()  # 设置模型为训练模式
        cur_epochs += 1  # 当前的迭代次数加1
        for (images, labels) in train_loader:  # 将训练集进行遍历
            cur_itrs += 1  # 当前的迭代次数加1
            # 对图片和标签进行转换，将图片和标签转换成tensor
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            # 进行梯度清零，将图像放入模型中，计算损失，反向传播，更新参数
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()  # 将损失放置到CPU上去，再转换为numpy
            interval_loss += np_loss  # 将损失累加起来
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:   # 每10次打印一次损失
                interval_loss = interval_loss / 10  # 将损失除以10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:  # 每100次进行一次验证
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))  # 保存最新的模型
                print("validation...")
                model.eval()  # 设置模型为验证模式
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)  # 我们的参数传入到验证中，从而获得我们的评价指标得分和返回的样本
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']  # 将最好的得分赋值给best_score
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])  # 将验证的评价指标得分进行可视化，转变成表格
                    # 对ret_samples进行遍历，将图片，标签，预测结果进行可视化
                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()  # 设置模型为训练模式
            scheduler.step()  # 更新学习率

            if cur_itrs >= opts.total_itrs:
                return  # 当当前的迭代次数大于等于总的迭代次数时，就返回


if __name__ == '__main__':
    main()
