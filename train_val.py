# -*- coding:utf-8 -*-
# @time :2022/2/23
# @author :Ziwei.zhan
# @Emial : 18908384271@163.com
import torch
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import train_dataloader,train_datasets, val_datasets, val_dataloader
import cfg
from utils import adjust_learning_rate_cosine, adjust_learning_rate_step
import warnings
warnings.warn("deprecated", DeprecationWarning) 
warnings.filterwarnings("ignore") 
##创建训练模型参数保存的文件夹
save_folder = cfg.SAVE_FOLDER + cfg.model_name
os.makedirs(save_folder, exist_ok=True)

def test():
    model.eval()
    total_correct = 0
    val_iter = iter(val_dataloader)
    max_iter = len(val_dataloader)
    for iteration in range(max_iter):
        try:
            images, labels = next(val_iter)
        except:
            continue
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
            out = model(images)
            prediction = torch.max(out, 1)[1]
            correct = (prediction == labels).sum()
            total_correct += correct
            print('Iteration: {}/{}'.format(iteration, max_iter), 'ACC: %.3f' %(correct.float()/batch_size))
    print('All ACC: %.3f'%(total_correct.float()/(len(val_dataloader)* batch_size)))
    return total_correct.float()/(len(val_dataloader)* batch_size)
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    return model

#####构建模型
if not cfg.RESUME_EPOCH:
    print('****** 训练中 {} ****** '.format(cfg.model_name))
    print('****** 读取预训练模型 ****** ')
    if not cfg.model_name.startswith('efficientnet'):
        if cfg.model_name == 'vit'or cfg.model_name == 'deepvit':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size=cfg.image_size,patch_size=cfg.patch_size,num_classes=cfg.NUM_CLASSES,dim=cfg.dim, depth=cfg.depth,heads=cfg.heads,mlp_dim=cfg.mlp_dim,dropout=cfg.dropout,emb_dropout=cfg.emb_dropout)
        elif cfg.model_name == 'cait':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size=cfg.image_size,patch_size=cfg.patch_size,num_classes=cfg.NUM_CLASSES,dim=cfg.dim, depth=cfg.depth,cls_depth=cfg.cls_depth,heads=cfg.heads,mlp_dim=cfg.mlp_dim,dropout=cfg.dropout,emb_dropout=cfg.emb_dropout,layer_dropout=cfg.layer_dropout)
        elif cfg.model_name == 't2tvit':
            model = cfg.MODEL_NAMES[cfg.model_name](dim=cfg.t2tvit_dim,image_size=cfg.t2tvit_image_size,depth=cfg.t2tvit_depth,heads=cfg.t2tvit_heads,mlp_dim=cfg.t2tvit_mlp_dim,num_classes=cfg.NUM_CLASSES,t2t_layers=cfg.t2tvit_t2t_layers)
        elif cfg.model_name == 'cct':
            model = cfg.MODEL_NAMES[cfg.model_name](img_size=cfg.cct_img_size,
                                                    embedding_dim=cfg.cct_embedding_dim,
                                                    n_conv_layers=cfg.cct_n_conv_layers,
                                                    kernel_size=cfg.cct_kernel_size,
                                                    stride=cfg.cct_stride,
                                                    padding=cfg.cct_padding,
                                                    pooling_kernel_size=cfg.cct_pooling_kernel_size,
                                                    pooling_stride=cfg.cct_pooling_stride,
                                                    pooling_padding=cfg.cct_pooling_padding,
                                                    num_layers=cfg.cct_num_layers,
                                                    num_heads=cfg.cct_num_heads,
                                                    mlp_radio=cfg.cct_mlp_radio,
                                                    num_classes=cfg.cct_num_classes,
                                                    positional_embedding=cfg.cct_positional_embedding)
        elif cfg.model_name == 'crossvit':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size = cfg.crossvit_image_size,
                                                    num_classes = cfg.crossvit_num_classes,
                                                    depth = cfg.crossvit_depth,  
                                                    sm_dim = cfg.crossvit_sm_dim,          
                                                    sm_patch_size = cfg.crossvit_sm_patch_size,      
                                                    sm_enc_depth = cfg.crossvit_sm_enc_depth,       
                                                    sm_enc_heads = cfg.crossvit_sm_enc_heads,       
                                                    sm_enc_mlp_dim = cfg.crossvit_sm_enc_mlp_dim,   
                                                    lg_dim = cfg.crossvit_lg_dim,          
                                                    lg_patch_size = cfg.crossvit_lg_patch_size,     
                                                    lg_enc_depth = cfg.crossvit_lg_enc_depth,       
                                                    lg_enc_heads = cfg.crossvit_lg_enc_heads,       
                                                    lg_enc_mlp_dim = cfg.crossvit_lg_enc_mlp_dim, 
                                                    cross_attn_depth = cfg.crossvit_cross_attn_depth,  
                                                    cross_attn_heads = cfg.crossvit_cross_attn_heads,
                                                    dropout = cfg.crossvit_dropout,
                                                    emb_dropout = cfg.crossvit_emb_dropout)
        elif cfg.model_name == 'pit':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size = cfg.pit_image_size,
                                                    patch_size = cfg.pit_patch_size,
                                                    dim = cfg.pit_dim,
                                                    num_classes = cfg.pit_num_classes,
                                                    depth = cfg.pit_depth, 
                                                    heads = cfg.pit_heads,
                                                    mlp_dim = cfg.pit_mlp_dim,
                                                    dropout = cfg.pit_dropout,
                                                    emb_dropout = cfg.pit_emb_dropout)
        elif cfg.model_name == 'levit':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size = cfg.levit_image_size,
                                                    num_classes = cfg.levit_num_classes,
                                                    stages = cfg.levit_stages,  
                                                    dim = cfg.levit_dim,  
                                                    depth = cfg.levit_depth,              
                                                    heads = cfg.levit_heads,     
                                                    mlp_mult = cfg.levit_mlp_mult,
                                                    dropout = cfg.levit_dropout)
        elif cfg.model_name == 'cvt':
            model = cfg.MODEL_NAMES[cfg.model_name](num_classes = cfg.cvt_num_classes,
                                                    s1_emb_dim = cfg.cvt_s1_emb_dim,        # stage 1 - dimension
                                                    s1_emb_kernel = cfg.cvt_s1_emb_kernel,      # stage 1 - conv kernel
                                                    s1_emb_stride = cfg.cvt_s1_emb_stride,      # stage 1 - conv stride
                                                    s1_proj_kernel = cfg.cvt_s1_proj_kernel,     # stage 1 - attention ds-conv kernel size
                                                    s1_kv_proj_stride = cfg.cvt_s1_kv_proj_stride,  # stage 1 - attention key / value projection stride
                                                    s1_heads = cfg.cvt_s1_heads,           # stage 1 - heads
                                                    s1_depth = cfg.cvt_s1_depth,           # stage 1 - depth
                                                    s1_mlp_mult = cfg.cvt_s1_mlp_mult,        # stage 1 - feedforward expansion factor
                                                    s2_emb_dim = cfg.cvt_s2_emb_dim,       # stage 2 - (same as above)
                                                    s2_emb_kernel = cfg.cvt_s2_emb_kernel,
                                                    s2_emb_stride = cfg.cvt_s2_emb_stride,
                                                    s2_proj_kernel = cfg.cvt_s2_proj_kernel,
                                                    s2_kv_proj_stride = cfg.cvt_s2_kv_proj_stride,
                                                    s2_heads = cfg.cvt_s2_heads,
                                                    s2_depth = cfg.cvt_s2_depth,
                                                    s2_mlp_mult = cfg.cvt_s2_mlp_mult,
                                                    s3_emb_dim = cfg.cvt_s3_emb_dim,       # stage 3 - (same as above)
                                                    s3_emb_kernel = cfg.cvt_s3_emb_kernel,
                                                    s3_emb_stride = cfg.cvt_s3_emb_stride,
                                                    s3_proj_kernel = cfg.cvt_s3_proj_kernel,
                                                    s3_kv_proj_stride = cfg.cvt_s3_kv_proj_stride,
                                                    s3_heads = cfg.cvt_s3_heads,
                                                    s3_depth = cfg.cvt_s3_depth,
                                                    s3_mlp_mult = cfg.cvt_s3_mlp_mult,
                                                    dropout = cfg.cvt_dropout)
        elif cfg.model_name == 'twinssvt':
            model = cfg.MODEL_NAMES[cfg.model_name](num_classes = cfg.twinssvt_num_classes,       # number of output classes
                                                    s1_emb_dim = cfg.twinssvt_s1_emb_dim,          # stage 1 - patch embedding projected dimension
                                                    s1_patch_size = cfg.twinssvt_s1_patch_size,        # stage 1 - patch size for patch embedding
                                                    s1_local_patch_size = cfg.twinssvt_s1_local_patch_size,  # stage 1 - patch size for local attention
                                                    s1_global_k = cfg.twinssvt_s1_global_k,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
                                                    s1_depth = cfg.twinssvt_s1_depth,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
                                                    s2_emb_dim = cfg.twinssvt_s2_emb_dim,         # stage 2 (same as above)
                                                    s2_patch_size = cfg.twinssvt_s2_patch_size,
                                                    s2_local_patch_size = cfg.twinssvt_s2_local_patch_size,
                                                    s2_global_k = cfg.twinssvt_s2_global_k,
                                                    s2_depth = cfg.twinssvt_s2_depth,
                                                    s3_emb_dim = cfg.twinssvt_s3_emb_dim,         # stage 3 (same as above)
                                                    s3_patch_size = cfg.twinssvt_s3_patch_size,
                                                    s3_local_patch_size = cfg.twinssvt_s3_local_patch_size,
                                                    s3_global_k = cfg.twinssvt_s3_global_k,
                                                    s3_depth = cfg.twinssvt_s3_depth,
                                                    s4_emb_dim = cfg.twinssvt_s4_emb_dim,         # stage 4 (same as above)
                                                    s4_patch_size = cfg.twinssvt_s4_patch_size,
                                                    s4_local_patch_size = cfg.twinssvt_s4_local_patch_size,
                                                    s4_global_k = cfg.twinssvt_s4_global_k,
                                                    s4_depth = cfg.twinssvt_s4_depth,
                                                    peg_kernel_size = cfg.twinssvt_peg_kernel_size,      # positional encoding generator kernel size
                                                    dropout = cfg.twinssvt_dropout  )

        elif cfg.model_name == 'regionvit':
            model = cfg.MODEL_NAMES[cfg.model_name](dim = cfg.regionvit_dim,      # tuple of size 4, indicating dimension at each stage
                                                    depth = cfg.regionvit_depth,           # depth of the region to local transformer at each stage
                                                    window_size = cfg.regionvit_window_size,                # window size, which should be either 7 or 14
                                                    num_classes = cfg.regionvit_num_classes,             # number of output classes
                                                    tokenize_local_3_conv = cfg.regionvit_tokenize_local_3_conv,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
                                                    use_peg = cfg.regionvit_use_peg )
        elif cfg.model_name == 'crossformer':
            model = cfg.MODEL_NAMES[cfg.model_name](num_classes = cfg.crossformer_num_classes,                # number of output classes
                                                    dim = cfg.crossformer_dim,         # dimension at each stage
                                                    depth = cfg.crossformer_depth,              # depth of transformer at each stage
                                                    global_window_size = cfg.crossformer_global_window_size, # global window sizes at each stage
                                                    local_window_size = cfg.crossformer_local_window_size,  )
        elif cfg.model_name == 'nest':
            model = cfg.MODEL_NAMES[cfg.model_name](image_size = cfg.nest_image_size,
                                                    patch_size = cfg.nest_patch_size,
                                                    dim = cfg.nest_dim,
                                                    heads = cfg.nest_heads,
                                                    num_hierarchies = cfg.nest_num_hierarchies,        # number of hierarchies
                                                    block_repeats = cfg.nest_block_repeats,  # the number of transformer blocks at each heirarchy, starting from the bottom
                                                    num_classes = cfg.nest_num_classes)
        elif cfg.model_name == 'swintransformer':
            model = cfg.MODEL_NAMES[cfg.model_name](img_size=cfg.swin_img_size,
                                                    patch_size=cfg.swin_patch_size,
                                                    in_chans=cfg.swin_in_chans,
                                                    num_classes=cfg.swin_num_classes,
                                                    embed_dim=cfg.swin_embed_dim,
                                                    depths=cfg.swin_depths,
                                                    num_heads=cfg.swin_num_heads,
                                                    window_size=cfg.swin_window_size,
                                                    mlp_ratio=cfg.swin_mlp_ratio,
                                                    qkv_bias=cfg.swin_qkv_bias,
                                                    qk_scale=cfg.swin_qk_scale,
                                                    drop_rate=cfg.swin_drop_rate,
                                                    drop_path_rate=cfg.swin_drop_path_rate,
                                                    ape=cfg.swin_ape,
                                                    patch_norm=cfg.swin_patch_norm,
                                                    use_checkpoint=cfg.swin_use_checkpoint)
        elif cfg.model_name == 'swinmlp':
            model = cfg.MODEL_NAMES[cfg.model_name](img_size=cfg.swinmlp_img_size,
                                                    patch_size=cfg.swinmlp_patch_size,
                                                    in_chans=cfg.swinmlp_in_chans,
                                                    num_classes=cfg.swinmlp_num_classes,
                                                    embed_dim=cfg.swinmlp_embed_dim,
                                                    depths=cfg.swinmlp_depths,
                                                    num_heads=cfg.swinmlp_num_heads,
                                                    window_size=cfg.swinmlp_window_size,
                                                    mlp_ratio=cfg.swinmlp_mlp_ratio,
                                                    drop_rate=cfg.swinmlp_drop_rate,
                                                    drop_path_rate=cfg.swinmlp_drop_path_rate,
                                                    ape=cfg.swinmlp_ape,
                                                    patch_norm=cfg.swinmlp_patch_norm,
                                                    use_checkpoint=cfg.swinmlp_use_checkpoint)
        elif cfg.model_name == 'coatnet':
            model = cfg.MODEL_NAMES[cfg.model_name](img_size=cfg.coatnet_img_size,
                                                    in_channels=cfg.coatnet_in_channels,
                                                    num_blocks=cfg.coatnet_num_blocks,
                                                    channels=cfg.coatnet_channels,
                                                    num_classes=cfg.coatnet_num_classes,
                                                    block_types=cfg.coatnet_block_types)
        else:
            model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)
        # #冻结前边一部分层不训练
        ct = 0
        for child in model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        
    else:
        model = cfg.MODEL_NAMES[cfg.model_name](cfg.model_name,num_classes=cfg.NUM_CLASSES)
        
        c = 0
        for name, p in model.named_parameters():
            c += 1
            if c >=700:
                break
            p.requires_grad = False
        

if cfg.RESUME_EPOCH:
    print(' ******* Resume training from {}  epoch {} *********'.format(cfg.model_name, cfg.RESUME_EPOCH))
    model = load_checkpoint(os.path.join(save_folder, 'epoch_{}.pth'.format(cfg.RESUME_EPOCH)))

##进行多gpu的并行计算
if cfg.GPUS>1:
    print('****** using multiple gpus to training ********')
    model = nn.DataParallel(model,device_ids=list(range(cfg.GPUS)))
else:
    print('****** using single gpu to training ********')
print("...... Initialize the network done!!! .......")

###模型放置在gpu上进行计算
if torch.cuda.is_available():
    model.cuda()
##定义优化器与损失函数
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)

optimizer = optim.SGD(model.parameters(), lr=cfg.LR,
                      momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
lr = cfg.LR
batch_size = cfg.BATCH_SIZE
#每一个epoch含有多少个batch
max_batch = len(train_datasets)//batch_size
epoch_size = len(train_datasets) // batch_size

## 训练max_epoch个epoch
max_iter = cfg.MAX_EPOCH * epoch_size
start_iter = cfg.RESUME_EPOCH * epoch_size
epoch = cfg.RESUME_EPOCH

# cosine学习率调整
warmup_epoch = cfg.WARMUP
warmup_steps = warmup_epoch * epoch_size
global_step = 0

# step 学习率调整参数
stepvalues = (40 * epoch_size, 60 * epoch_size, 80 * epoch_size)

step_index = 0

nice = 0
for iteration in range(start_iter, max_iter):
    global_step += 1
    ##更新迭代器
    if iteration % epoch_size == 0:
        # create batch iterator
        batch_iterator = iter(train_dataloader)
        loss = 0
        epoch += 1
        if epoch > 1:
            pass
        temp = test()
        if nice < temp:
            nice = temp
        ###保存模型
        model.train()
        if epoch % 10 == 0 and epoch > 0:
            if cfg.GPUS > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'epoch_{}.pth'.format(epoch)))
    if iteration in stepvalues:
        step_index += 1
    lr = adjust_learning_rate_step(optimizer, cfg.LR, 0.1, epoch, step_index, iteration, epoch_size)

    ## 调整学习率
    # lr = adjust_learning_rate_cosine(optimizer, global_step=global_step,
    #                           learning_rate_base=cfg.LR,
    #                           total_steps=max_iter,
    #                           warmup_steps=warmup_steps)
    ## 获取image 和 label
    try:
        images, labels = next(batch_iterator)
    except:
        continue
    if torch.cuda.is_available():
        images, labels = images.cuda(), labels.cuda()
    out = model(images)
    #转换标签格式

    #out = torch.tensor(out)
    labels = labels.to(torch.int64)
    
    loss = criterion(out, labels)

    # 添加所有参数梯度
    loss.requires_grad_(True)
    
    # 清空梯度信息，否则在每次进行反向传播时都会累加
    optimizer.zero_grad()  
    loss.backward()
    optimizer.step()
    prediction = torch.max(out, 1)[1]
    train_correct = (prediction == labels).sum()
    ##这里得到的train_correct是一个longtensor型，需要转换为float
    # print(train_correct.type())
    train_acc = (train_correct.float()) / batch_size
    if iteration % 10 == 0:
        print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
              + '|| Totel iter ' + repr(iteration) + ' || Loss: %.6f||' % (loss.item()) + 'ACC: %.3f ||' %(train_acc * 100) + 'LR: %.8f' % (lr))
print(nice)