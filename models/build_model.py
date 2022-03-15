# -*- coding:utf-8 -*-
# @time :2022/2/23
# @author :Ziwei.zhan
# @Emial : 18908384271@163.com

from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch
import torchvision
from models import resnet101, densenet121, densenet169, resnet50, mobilenet_v2,resnet18,resnet34,resnet152,GoogLeNet,Inception3
from models import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl
from models import EfficientNet,squeezenet1_1,squeezenet1_0
from models import ResNeXt50, ResNeXt152
from models import mobilevit_xxs, mobilevit_xs, mobilevit_s
from models import ConvMixer
from models import densenet161,densenet201,shufflenet_v2_x2_0,shufflenet_v2_x1_5,shufflenet_v2_x1_0,shufflenet_v2_x0_5
from models import ViT,vgg16,vgg19,VGG,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16_bn,vgg19_bn,AlexNet
from models import DeepViT,CaiT,T2TViT
from models import CCT,ghostnet,MobileNetV3_Large,MobileNetV3_Small
from models import LOCAL_PRETRAINED, model_urls,CoAtNet
from models import CrossViT,PiT,LeViT,CvT,TwinsSVT,RegionViT,CrossFormer,NesT,SwinTransformer,SwinMLP

def Coatnet(img_size,in_channels,num_blocks,channels,num_classes,block_types):
    model = CoAtNet(img_size, 
                    in_channels, 
                    num_blocks, 
                    channels, 
                    num_classes,
                    block_types)
    return model
def Swinmlp(img_size,patch_size,in_chans,num_classes,embed_dim,depths,num_heads,window_size,mlp_ratio,drop_rate,drop_path_rate,ape,patch_norm,use_checkpoint):
    model = SwinMLP(img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    num_classes=num_classes,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    ape=ape,
                    patch_norm=patch_norm,
                    use_checkpoint=use_checkpoint)
    return model

def Swintransformer(img_size,patch_size,in_chans,num_classes,embed_dim,depths,num_heads,window_size,mlp_ratio,qkv_bias,qk_scale,drop_rate,drop_path_rate,ape,patch_norm,use_checkpoint):
    model = SwinTransformer(img_size=img_size,
                            patch_size=patch_size,
                            in_chans=in_chans,
                            num_classes=num_classes,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate,
                            ape=ape,
                            patch_norm=patch_norm,
                            use_checkpoint=use_checkpoint)
    return model

def Mobilenetv3_small(num_classes, test=False):
    model = MobileNetV3_Small()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Mobilenetv3_large(num_classes, test=False):
    model = MobileNetV3_Large()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Ghostnet(num_classes, test=False):
    model = ghostnet()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Alexnet(num_classes, test=False):
    model = AlexNet()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg(num_classes, test=False):
    model = VGG()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Vgg11(num_classes, test=False):
    model = vgg11()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg11_bn(num_classes, test=False):
    model = vgg11_bn()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg13(num_classes, test=False):
    model = vgg13()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg13_bn(num_classes, test=False):
    model = vgg13_bn()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg16_bn(num_classes, test=False):
    model = vgg16_bn()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg19_bn(num_classes, test=False):
    model = vgg19_bn()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg16(num_classes, test=False):
    model = vgg16()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Vgg19(num_classes, test=False):
    model = vgg19()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def shufflenet_v2_x05(num_classes, test=False):
    model = shufflenet_v2_x0_5()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def shufflenet_v2_x10(num_classes, test=False):
    model = shufflenet_v2_x1_0()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def shufflenet_v2_x15(num_classes, test=False):
    model = shufflenet_v2_x1_5()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def shufflenet_v2_x2(num_classes, test=False):
    model = shufflenet_v2_x2_0()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def squeezenet11(num_classes, test=False):
    model = squeezenet1_1(num_classes)
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    #fc_features = model.fc.in_features
    #model.fc = nn.Linear(fc_features, num_classes)
    return model

def squeezenet10(num_classes, test=False):
    model = squeezenet1_0(num_classes)
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    #fc_features = model.fc.in_features
    #model.fc = nn.Linear(fc_features, num_classes)
    return model

def Nest(image_size,patch_size,dim,heads,num_hierarchies,block_repeats,num_classes):
    model = NesT(
        image_size = image_size,
        patch_size = patch_size,
        dim = dim,
        heads = heads,
        num_hierarchies = num_hierarchies,        # number of hierarchies
        block_repeats = block_repeats,  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes = num_classes
    )
    return model

def Crossformer(num_classes,dim,depth,global_window_size,local_window_size):
    model = CrossFormer(
        num_classes = num_classes,                # number of output classes
        dim = dim,         # dimension at each stage
        depth = depth,              # depth of transformer at each stage
        global_window_size = global_window_size, # global window sizes at each stage
        local_window_size = local_window_size,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    )
    return model


def Regionvit(dim,depth,window_size,num_classes,tokenize_local_3_conv,use_peg):
    model = RegionViT(
        dim = dim,      # tuple of size 4, indicating dimension at each stage
        depth = depth,           # depth of the region to local transformer at each stage
        window_size = window_size,                # window size, which should be either 7 or 14
        num_classes = num_classes,             # number of output classes
        tokenize_local_3_conv = tokenize_local_3_conv,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
        use_peg = use_peg,                # whether to use positional generating module. they used this for object detection for a boost in performance
    )
    return model

def Twinssvt(num_classes ,s1_emb_dim ,s1_patch_size ,s1_local_patch_size ,s1_global_k ,s1_depth ,s2_emb_dim ,s2_patch_size ,s2_local_patch_size ,s2_global_k ,s2_depth ,s3_emb_dim ,s3_patch_size ,s3_local_patch_size ,s3_global_k ,s3_depth ,s4_emb_dim ,s4_patch_size ,s4_local_patch_size ,s4_global_k ,s4_depth ,peg_kernel_size ,dropout):
    model = TwinsSVT(
        num_classes = num_classes,       # number of output classes
        s1_emb_dim = s1_emb_dim,          # stage 1 - patch embedding projected dimension
        s1_patch_size = s1_patch_size,        # stage 1 - patch size for patch embedding
        s1_local_patch_size = s1_local_patch_size,  # stage 1 - patch size for local attention
        s1_global_k = s1_global_k,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        s1_depth = s1_depth,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        s2_emb_dim = s2_emb_dim,         # stage 2 (same as above)
        s2_patch_size = s2_patch_size,
        s2_local_patch_size = s2_local_patch_size,
        s2_global_k = s2_global_k,
        s2_depth = s2_depth,
        s3_emb_dim = s3_emb_dim,         # stage 3 (same as above)
        s3_patch_size = s3_patch_size,
        s3_local_patch_size = s3_local_patch_size,
        s3_global_k = s3_global_k,
        s3_depth = s3_depth,
        s4_emb_dim = s4_emb_dim,         # stage 4 (same as above)
        s4_patch_size = s4_patch_size,
        s4_local_patch_size = s4_local_patch_size,
        s4_global_k = s4_global_k,
        s4_depth = s4_depth,
        peg_kernel_size = peg_kernel_size,      # positional encoding generator kernel size
        dropout = dropout              # dropout
    )
    return model

def Cvt(num_classes,s1_emb_dim,s1_emb_kernel,s1_emb_stride,s1_proj_kernel,s1_kv_proj_stride,s1_heads,s1_depth,s1_mlp_mult ,s2_emb_dim ,s2_emb_kernel ,s2_emb_stride ,s2_proj_kernel ,s2_kv_proj_stride ,s2_heads ,s2_depth ,s2_mlp_mult ,s3_emb_dim ,s3_emb_kernel ,s3_emb_stride ,s3_proj_kernel ,s3_kv_proj_stride ,s3_heads ,s3_depth ,s3_mlp_mult ,dropout ):
    model = CvT(
        num_classes = num_classes,
        s1_emb_dim = s1_emb_dim,        # stage 1 - dimension
        s1_emb_kernel = s1_emb_kernel,      # stage 1 - conv kernel
        s1_emb_stride = s1_emb_stride,      # stage 1 - conv stride
        s1_proj_kernel = s1_proj_kernel,     # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride = s1_kv_proj_stride,  # stage 1 - attention key / value projection stride
        s1_heads = s1_heads,           # stage 1 - heads
        s1_depth = s1_depth,           # stage 1 - depth
        s1_mlp_mult = s1_mlp_mult,        # stage 1 - feedforward expansion factor
        s2_emb_dim = s2_emb_dim,       # stage 2 - (same as above)
        s2_emb_kernel = s2_emb_kernel,
        s2_emb_stride = s2_emb_stride,
        s2_proj_kernel = s2_proj_kernel,
        s2_kv_proj_stride = s2_kv_proj_stride,
        s2_heads = s2_heads,
        s2_depth = s2_depth,
        s2_mlp_mult = s2_mlp_mult,
        s3_emb_dim = s3_emb_dim,       # stage 3 - (same as above)
        s3_emb_kernel = s3_emb_kernel,
        s3_emb_stride = s3_emb_stride,
        s3_proj_kernel = s3_proj_kernel,
        s3_kv_proj_stride = s3_kv_proj_stride,
        s3_heads = s3_heads,
        s3_depth = s3_depth,
        s3_mlp_mult = s3_mlp_mult,
        dropout = dropout
    )
    return model

def Levit(image_size ,num_classes ,stages ,dim ,depth ,heads ,mlp_mult ,dropout ):
    model = LeViT(
    image_size = image_size,
    num_classes = num_classes,
    stages = stages,       
    dim = dim, 
    depth = depth,         
    heads = heads,     
    mlp_mult = mlp_mult,
    dropout = dropout
    )
    return model


def Pit(image_size,patch_size,dim,num_classes,depth,heads,mlp_dim,dropout,emb_dropout):
    model = PiT(
    image_size = image_size,
    patch_size = patch_size,
    dim = dim,
    num_classes = num_classes,
    depth = depth,   
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
    )
    return model
def Crossvit(image_size,num_classes,depth,sm_dim,sm_patch_size,sm_enc_depth,sm_enc_heads,sm_enc_mlp_dim,lg_dim,lg_patch_size,lg_enc_depth,lg_enc_heads,lg_enc_mlp_dim,cross_attn_depth,cross_attn_heads,dropout,emb_dropout):
    model = CrossViT(
    image_size = image_size,
    num_classes = num_classes,
    depth = depth,     
    sm_dim = sm_dim,         
    sm_patch_size = sm_patch_size, 
    sm_enc_depth = sm_enc_depth,        
    sm_enc_heads = sm_enc_heads,       
    sm_enc_mlp_dim = sm_enc_mlp_dim,   
    lg_dim = lg_dim,         
    lg_patch_size = lg_patch_size,     
    lg_enc_depth = lg_enc_depth,    
    lg_enc_heads = lg_enc_heads,        
    lg_enc_mlp_dim = lg_enc_mlp_dim,   
    cross_attn_depth = cross_attn_depth,    
    cross_attn_heads = cross_attn_heads,    
    dropout = dropout,
    emb_dropout = emb_dropout
    )
    return model

def Cct(img_size,embedding_dim,n_conv_layers,kernel_size,stride,padding,pooling_kernel_size,pooling_stride,pooling_padding,num_layers,num_heads,mlp_radio,num_classes,positional_embedding):
    model = CCT(
    img_size=img_size,
    embedding_dim=embedding_dim,
    n_conv_layers=n_conv_layers,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    pooling_kernel_size=pooling_kernel_size,
    pooling_stride=pooling_stride,
    pooling_padding=pooling_padding,
    num_layers=num_layers,
    num_heads=num_heads,
    mlp_radio=mlp_radio,
    num_classes=num_classes,
    positional_embedding=positional_embedding,
    )
    return model

def T2tvit(dim,image_size,depth,heads,mlp_dim,num_classes,t2t_layers):
    model = T2TViT(
    dim = dim,
    image_size = image_size,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    num_classes = num_classes,
    t2t_layers = t2t_layers # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )
    return model

def Cait(image_size,patch_size,num_classes,dim, depth,cls_depth,heads,mlp_dim,dropout,emb_dropout,layer_dropout,test=False):
    model = CaiT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = dim,
    depth = depth,             
    cls_depth = cls_depth, 
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout,
    layer_dropout = layer_dropout
    )
    return model

def Deepvit(image_size,patch_size,num_classes,dim, depth,heads,mlp_dim,dropout,emb_dropout,test=False):
    model = DeepViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = dim,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
    )
    return model

def Vit(image_size,patch_size,num_classes,dim, depth,heads,mlp_dim,dropout,emb_dropout,test=False):
    model = ViT(
    image_size = image_size,
    patch_size = patch_size,
    num_classes = num_classes,
    dim = dim,
    depth = depth,
    heads = heads,
    mlp_dim = mlp_dim,
    dropout = dropout,
    emb_dropout = emb_dropout
    )
    return model

def Convmixer(num_classes, test=False):
    model = ConvMixer(num_classes)
    return model

def Mobilevit_xxs(num_classes, test=False):
    model = mobilevit_xxs()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Mobilevit_xs(num_classes, test=False):
    model = mobilevit_xs()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Mobilevit_s(num_classes, test=False):
    model = mobilevit_s()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet18(num_classes, test=False):
    model = resnet18()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet34(num_classes, test=False):
    model = resnet34()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet152(num_classes, test=False):
    model = resnet152()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnet50(num_classes, test=False):
    model = resnet50()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Googlenet(num_classes, test=False):
    model = GoogLeNet()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def inception3(num_classes, test=False):
    model = Inception3()
    # if not test:
        # if LOCAL_PRETRAINED['resnet50'] == None:
            # state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
        # else:
            # state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet50'])
        # model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model



def Resnext50(num_classes, test=False):
    model = ResNeXt50()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model

def Resnext152(num_classes, test=False):
    model = ResNeXt152()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnet101(num_classes, test=False):
    model = resnet101()
    '''
    if not test:
        if LOCAL_PRETRAINED['resnet101'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnet101'])
        model.load_state_dict(state_dict)
    '''
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x8d(num_classes, test=False):
    model = resnext101_32x8d_wsl()
    if not test:
        if LOCAL_PRETRAINED['resnext101_32x8d'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x8d'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x16d(num_classes, test=False):
    model = resnext101_32x16d_wsl()
    if not test:
        if LOCAL_PRETRAINED['resnext101_32x16d'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnext101_32x16d'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x16d'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x32d(num_classes, test=False):
    model = resnext101_32x32d_wsl()
    if not test:
        if LOCAL_PRETRAINED['resnext101_32x32d'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnext101_32x32d'], progress=True)
        else:
            print(LOCAL_PRETRAINED['resnext101_32x32d'])
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x32d'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Resnext101_32x48d(num_classes, test=False):
    model = resnext101_32x48d_wsl()
    if not test:
        if LOCAL_PRETRAINED['resnext101_32x48d'] == None:
            state_dict = load_state_dict_from_url(model_urls['resnext101_32x48d'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['resnext101_32x48d'])
        model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)
    return model


def Densenet121(num_classes, test=False):
    model = densenet121()
    '''
    if not test:
        if LOCAL_PRETRAINED['densenet121'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet121'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet121'])

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k,v in state_dict.items():
            # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
            # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
            # 直接利用if语句筛选不一致的键
            ### 修正键值的不对应
            if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    '''
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def Densenet161(num_classes, test=True):
    model = densenet161()
    '''
    if not test:
        if LOCAL_PRETRAINED['densenet161'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet161'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet161'])

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k,v in state_dict.items():
            # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
            # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
            # 直接利用if语句筛选不一致的键
            ### 修正键值的不对应
            if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    '''
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def Densenet169(num_classes, test=False):
    model = densenet169()
    '''
    if not test:
        if LOCAL_PRETRAINED['densenet169'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet169'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet169'])

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k,v in state_dict.items():
            # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
            # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
            # 直接利用if语句筛选不一致的键
            ### 修正键值的不对应
            if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    '''
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model


def Densenet201(num_classes, test=False):
    model = densenet201()
    '''
    if not test:
        if LOCAL_PRETRAINED['densenet201'] == None:
            state_dict = load_state_dict_from_url(model_urls['densenet201'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['densenet201'])

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k,v in state_dict.items():
            # print(k)  #打印预训练模型的键，发现与网络定义的键有一定的差别，因而需要将键值进行对应的更改，将键值分别对应打印出来就可以看出不同，根据不同进行修改
            # torchvision中的网络定义，采用了正则表达式，来更改键值，因为这里简单，没有再去构建正则表达式
            # 直接利用if语句筛选不一致的键
            ### 修正键值的不对应
            if k.split('.')[0] == 'features' and (len(k.split('.')))>4:
                k = k.split('.')[0]+'.'+k.split('.')[1]+'.'+k.split('.')[2]+'.'+k.split('.')[-3] + k.split('.')[-2] +'.'+k.split('.')[-1]
            # print(k)
            else:
                pass
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    '''
    fc_features = model.classifier.in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def Mobilenetv2(num_classes, test=False):
    model = mobilenet_v2()
    '''
    if not test:
        if LOCAL_PRETRAINED['moblienetv2'] == None:
            state_dict = load_state_dict_from_url(model_urls['moblienetv2'], progress=True)
        else:
            state_dict = state_dict = torch.load(LOCAL_PRETRAINED['moblienetv2'])
        model.load_state_dict(state_dict)
    print(model.state_dict().keys())
    '''
    fc_features = model.classifier[1].in_features
    model.classifier = nn.Linear(fc_features, num_classes)
    return model

def Efficientnet(model_name, num_classes, test = False):
    '''
    model_name :'efficientnet-b0', 'efficientnet-b1-7'
    '''
    model = EfficientNet.from_name(model_name)
    if not test:
        if LOCAL_PRETRAINED[model_name] == None:
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        else:
            state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        model.load_state_dict(state_dict)
    fc_features = model._fc.in_features
    model._fc = nn.Linear(fc_features, num_classes)
    return model




