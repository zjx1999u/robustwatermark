from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class  HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out
'''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
'''
"""
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    layer1.1.conv2
    layer4.1.conv2
"""
def  get_feas_by_hook(model):
    fea_hooks = []
    for name, layer in model.named_modules():
        if '.1.conv2' in name:
            cur_hook = HookTool()
            layer.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
            #print('shape of  feature is:', fea_hooks[0].fea.shape,'**')
    return fea_hooks

def pairwise_euclidean_distance(a, b):#a,b:tensor
    ba = a.shape[0]
    bb = b.shape[0]
    sqr_norm_a = torch.reshape(torch.sum(torch.pow(a,2),axis=1),(1,ba))
    sqr_norm_b = torch.reshape(torch.sum(torch.pow(b,2),axis=1),(bb,1))
    inner_prod = torch.matmul(b,torch.transpose(a,0,1))
    tile1 = sqr_norm_a.repeat(bb, 1)
    tile2 = sqr_norm_b.repeat(1, ba)
    
    return tile1 + tile2 - 2 * inner_prod

def snnl_loss(out,targets,temprature):
    out1 = torch.reshape(out,(out.shape[0],-1))#变成二维向量
    out2 = pairwise_euclidean_distance(out1,out1)

    exp_distance = torch.exp(-(out2 / (temprature + 1e-7)))
    f = exp_distance - torch.eye(out1.shape[0]).to('cuda')
    f[f >= 1] = 1
    f[f <= 0] = 0#取值范围映射到0-1

    pick_probability = f / (1e-7 + torch.sum(f, dim=1, keepdim=True))#dim取值存疑

    targets_shape = targets.shape[0]
    same_label_mask = torch.eq(targets, targets.reshape(targets_shape, 1))
    same_label_mask = same_label_mask.float()
    masked_pick_probability = pick_probability * same_label_mask
    sum_masked_probability = torch.sum(masked_pick_probability, axis=1)
    loss = torch.mean(-torch.log(1e-7 + sum_masked_probability))

    return loss
#        ba = functional.shape(a)[0]
 #               bb = functional.shape(b)[0]
 #               sqr_norm_a = functional.reshape(functional.sum(functional.pow(a, 2), axis=1), 1, ba)#行向量内部相加
#                sqr_norm_b = functional.reshape(functional.sum(functional.pow(b, 2), axis=1), bb, 1)
 #               inner_prod = functional.matmul(b, functional.transpose(a))
 #               tile1 = functional.tile(sqr_norm_a, bb, 1)
  #              tile2 = functional.tile(sqr_norm_b, 1, ba)
   #             return tile1 + tile2 - 2 * inner_prod

