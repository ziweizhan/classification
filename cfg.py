# -*- coding:utf-8 -*-
# @time :2022/2/23
# @author :Ziwei.zhan
# @Emial : 18908384271@163.com

import os
path = os.getcwd()
#*******************************************通用参数设置*******************************************
#定义路径
BASE = path + '//data//'
##数据集的类别
NUM_CLASSES = 4
#训练时batch的大小
BATCH_SIZE = 16
#网络默认输入图像的大小
INPUT_SIZE = 512
#训练最多的epoch
MAX_EPOCH = 100
# 使用gpu的数目
GPUS = 1
# 从第几个epoch开始resume训练，如果为0，从头开始
RESUME_EPOCH = 0
# 学习率衰减
WEIGHT_DECAY = 5e-4
# 动量
MOMENTUM = 0.9
# 初始学习率
LR = 0.01
# 余弦学习率调整
WARMUP = 5
# 采用的模型名称
model_name = 'vgg11_bn'
# 训练好模型的保存位置
SAVE_FOLDER = BASE + 'weights/'
#数据集的存放位置
TRAIN_LABEL_DIR =BASE + 'train.txt'     
VAL_LABEL_DIR = BASE + 'val.txt'
TEST_LABEL_DIR = BASE + 'test.txt'
##训练完成，权重文件的保存路径,默认保存在trained_model下
TRAINED_MODEL = BASE + 'weights/resnext101_32x32d/epoch_40.pth'
#*******************************************单独算法参数设置**************************************** ***
# vit,Deepvit和Cait 模型参数
image_size  = INPUT_SIZE
patch_size = 8
dim = 1024
depth = 6
heads = 16
mlp_dim = 2048
dropout = 0.1
emb_dropout = 0.1
# Cait专用参数
cls_depth = 2
layer_dropout = 0.05

#T2tvit模型参数
t2tvit_dim = 512
t2tvit_image_size = INPUT_SIZE
t2tvit_depth = 5
t2tvit_heads = 8
t2tvit_mlp_dim = 512
t2tvit_t2t_layers = ((7, 4), (3, 2), (3, 2))

#cct模型参数：
cct_img_size=INPUT_SIZE
cct_embedding_dim=384
cct_n_conv_layers=2
cct_kernel_size=7
cct_stride=2
cct_padding=3
cct_pooling_kernel_size=3
cct_pooling_stride=2
cct_pooling_padding=1
cct_num_layers=14
cct_num_heads=6
cct_mlp_radio=3.
cct_num_classes=NUM_CLASSES
cct_positional_embedding='learnable'

#crossvit模型参数：
crossvit_image_size = INPUT_SIZE
crossvit_num_classes = NUM_CLASSES
crossvit_depth = 4   
crossvit_sm_dim = 192         
crossvit_sm_patch_size = 16     
crossvit_sm_enc_depth = 2      
crossvit_sm_enc_heads = 8       
crossvit_sm_enc_mlp_dim = 2048 
crossvit_lg_dim = 384       
crossvit_lg_patch_size = 64    
crossvit_lg_enc_depth = 3     
crossvit_lg_enc_heads = 8      
crossvit_lg_enc_mlp_dim = 2048   
crossvit_cross_attn_depth = 2  
crossvit_cross_attn_heads = 8 
crossvit_dropout = 0.1
crossvit_emb_dropout = 0.1

#pit模型参数
pit_image_size = INPUT_SIZE
pit_patch_size = 14
pit_dim = 256
pit_num_classes = NUM_CLASSES
pit_depth = (3, 3, 3)
pit_heads = 16
pit_mlp_dim = 2048
pit_dropout = 0.1
pit_emb_dropout = 0.1

#levit
levit_image_size = INPUT_SIZE
levit_num_classes = NUM_CLASSES
levit_stages = 3
levit_dim = (256, 384, 512)
levit_depth = 4
levit_heads = (4, 6, 8)
levit_mlp_mult = 2
levit_dropout = 0.1

#cvt模型参数
cvt_num_classes = NUM_CLASSES 
cvt_s1_emb_dim = 64        # stage 1 - dimension
cvt_s1_emb_kernel = 7      # stage 1 - conv kernel
cvt_s1_emb_stride = 4      # stage 1 - conv stride
cvt_s1_proj_kernel = 3     # stage 1 - attention ds-conv kernel size
cvt_s1_kv_proj_stride = 2  # stage 1 - attention key / value projection stride
cvt_s1_heads = 1           # stage 1 - heads
cvt_s1_depth = 1           # stage 1 - depth
cvt_s1_mlp_mult = 4        # stage 1 - feedforward expansion factor
cvt_s2_emb_dim = 192       # stage 2 - (same as above)
cvt_s2_emb_kernel = 3
cvt_s2_emb_stride = 2
cvt_s2_proj_kernel = 3
cvt_s2_kv_proj_stride = 2
cvt_s2_heads = 3
cvt_s2_depth = 2
cvt_s2_mlp_mult = 4
cvt_s3_emb_dim = 384       # stage 3 - (same as above)
cvt_s3_emb_kernel = 3
cvt_s3_emb_stride = 2
cvt_s3_proj_kernel = 3
cvt_s3_kv_proj_stride = 2
cvt_s3_heads = 4
cvt_s3_depth = 10
cvt_s3_mlp_mult = 4
cvt_dropout = 0.

#twinssvt模型参数
twinssvt_num_classes = NUM_CLASSES       # number of output classes
twinssvt_s1_emb_dim = 64          # stage 1 - patch embedding projected dimension
twinssvt_s1_patch_size = 4        # stage 1 - patch size for patch embedding
twinssvt_s1_local_patch_size = 7  # stage 1 - patch size for local attention
twinssvt_s1_global_k = 7          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
twinssvt_s1_depth = 1             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
twinssvt_s2_emb_dim = 128         # stage 2 (same as above)
twinssvt_s2_patch_size = 2
twinssvt_s2_local_patch_size = 7
twinssvt_s2_global_k = 7
twinssvt_s2_depth = 1
twinssvt_s3_emb_dim = 256         # stage 3 (same as above)
twinssvt_s3_patch_size = 2
twinssvt_s3_local_patch_size = 7
twinssvt_s3_global_k = 7
twinssvt_s3_depth = 5
twinssvt_s4_emb_dim = 512         # stage 4 (same as above)
twinssvt_s4_patch_size = 2
twinssvt_s4_local_patch_size = 7
twinssvt_s4_global_k = 7
twinssvt_s4_depth = 4
twinssvt_peg_kernel_size = 3      # positional encoding generator kernel size
twinssvt_dropout = 0.    

#Regionvit模型参数
regionvit_dim = (64, 128, 256, 512)
regionvit_depth = (2, 2, 8, 2)
regionvit_window_size = 7
regionvit_num_classes = NUM_CLASSES
regionvit_tokenize_local_3_conv = False
regionvit_use_peg = False

#Crossformer模型参数
crossformer_num_classes = NUM_CLASSES                # number of output classes
crossformer_dim = (64, 128, 256, 512)         # dimension at each stage
crossformer_depth = (2, 2, 8, 2)              # depth of transformer at each stage
crossformer_global_window_size = (8, 4, 2, 1) # global window sizes at each stage
crossformer_local_window_size = 7  

#nest模型参数
nest_image_size = INPUT_SIZE
nest_patch_size = NUM_CLASSES
nest_dim = 96
nest_heads = 3
nest_num_hierarchies = 3
nest_block_repeats = (2, 2, 8)
nest_num_classes = 4

#swintransformer模型参数
swin_img_size = INPUT_SIZE
swin_patch_size=4
swin_in_chans=3
swin_num_classes= NUM_CLASSES
swin_embed_dim=96
swin_depths=[2, 2, 6, 2]
swin_num_heads=[3, 6, 12, 24]
swin_window_size=7
swin_mlp_ratio=4.
swin_qkv_bias=True
swin_qk_scale=None
swin_drop_rate=0.0
swin_drop_path_rate=0.1
swin_ape=False
swin_patch_norm=True
swin_use_checkpoint=False

#swimmlp模型参数
swinmlp_img_size=INPUT_SIZE
swinmlp_patch_size=4
swinmlp_in_chans=3
swinmlp_num_classes=NUM_CLASSES
swinmlp_embed_dim=96
swinmlp_depths=[2, 2, 6, 2]
swinmlp_num_heads=[3, 6, 12, 24]
swinmlp_window_size=7
swinmlp_mlp_ratio=4.
swinmlp_drop_rate=0.0
swinmlp_drop_path_rate=0.1
swinmlp_ape=False
swinmlp_patch_norm=True
swinmlp_use_checkpoint=False

#coatnet模型参数
coatnet_num_classes = NUM_CLASSES
coatnet_img_size = (224,224)
coatnet_in_channels = 3
coatnet_num_blocks = [2, 2, 3, 5, 2]
coatnet_channels = [64, 96, 192, 384, 768]  
coatnet_block_types=['C', 'T', 'T', 'T']    

from models import Resnet18,Resnet34,Resnet152,Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d
from models import Resnext50, Resnext152,Googlenet,inception3,Alexnet,Coatnet
from models import Mobilevit_xxs, Mobilevit_xs, Mobilevit_s,Swintransformer
from models import Convmixer,squeezenet11,squeezenet10,shufflenet_v2_x2,shufflenet_v2_x15,shufflenet_v2_x10,shufflenet_v2_x05
from models import Densenet161,Densenet201,Ghostnet,Mobilenetv3_small,Mobilenetv3_large
from models import Vit,Deepvit,Cait,T2tvit,Vgg16,Vgg19,Vgg,Vgg11,Vgg11_bn,Vgg13,Vgg13_bn,Vgg16_bn,Vgg19_bn
from models import Cct,Crossvit,Pit,Levit,Cvt,Twinssvt,Regionvit,Crossformer,Nest,Swinmlp


MODEL_NAMES = {
    'coatnet':Coatnet,
    'swinmlp':Swinmlp,
    'swintransformer':Swintransformer,
    'mobilenetv3_small':Mobilenetv3_small,
    'mobilenetv3_large':Mobilenetv3_large,
    'ghostnet':Ghostnet,
    'alexnet': Alexnet,
    'vgg11':Vgg11,
    'vgg11_bn':Vgg11_bn,
    'vgg13':Vgg13,
    'vgg13_bn':Vgg13_bn,
    'vgg-16':Vgg16,
    'vgg-16_bn':Vgg16_bn,
    'vgg-19':Vgg19,
    'vgg-19_bn':Vgg19_bn,
    'shufflenet_v2.05': shufflenet_v2_x05,
    'shufflenet_v2.10': shufflenet_v2_x10,
    'shufflenet_v2.15': shufflenet_v2_x15,
    'shufflenet_v2.2': shufflenet_v2_x2,
    'squeezenet10':squeezenet10,
    'squeezenet11':squeezenet11,
    'nest': Nest,
    'crossformer': Crossformer,
    'regionvit': Regionvit,
    'twinssvt': Twinssvt,
    'cvt': Cvt,
    'levit': Levit,
    'pit': Pit,
    'crossvit': Crossvit,
    'cct': Cct,
    'vit': Vit,
    'deepvit': Deepvit,
    'cait': Cait,
    't2tvit':T2tvit,
    'convmixer': Convmixer,
    'mobilevit_xxs': Mobilevit_xxs,
    'mobilevit_xs': Mobilevit_xs,
    'mobilevit_s': Mobilevit_s,
    'resnext50': Resnext50,
    'resnext152': Resnext152,
    'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet18': Resnet18,
    'resnet34': Resnet34,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'resnet152': Resnet152,
    'densenet121': Densenet121,
    'densenet161': Densenet161,
    'densenet169': Densenet169,
    'densenet201': Densenet201,
    'moblienetv2': Mobilenetv2,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet,
    'googlenet': Googlenet,
    'inceptionv3':inception3
}