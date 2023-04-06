import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from helpers.utils import progress_bar
import sys
import random
# Train function   Loss= (Lc+lamda *L2 cleandata)/1+lamda
from models.hooktool import *

def calculate_tensor_percentile(t: torch.tensor, q: float):

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def poison_data(x, y, sample_index):
    B, C, H, W = x.size()
    assert(C == 3)
    
    index = []
    init_index = []
    for i in range(B):
        if bool(sample_index[i]):
            index.append(i) #wm 后的图片id
        else:
            init_index.append(i)

    print(y[index])
    y[index] = 0
    
    
    h0, w0 = H-4, W-4
    for h, w in [(h0, w0), (h0-1, w0-1), (h0+1, w0+1), (h0-1, w0+1), (h0+1, w0-1)]:
        for j in range(C):
            x[index,j,h,w]=0
    return x, y, init_index, index


def poison_data_all(x, y):
    B, C, H, W = x.size()
    assert(C == 3)
    
    index = []
    init_index = []
    for i in range(B):
        if 1 :
            index.append(i) #wm 后的图片id
    y[index] = 0
    
    
    h0, w0 = H-4, W-4
    for h, w in [(h0, w0), (h0-1, w0-1), (h0+1, w0+1), (h0-1, w0+1), (h0+1, w0-1)]:
        for j in range(C):
            x[index,j,h,w]=0
    return x, y, init_index, index


def train(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
    
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def train_with_hook(epoch, net, criterion, optimizer, logfile, loader, device, sample_index,tune_all=True):
    print('\nEpoch: %d,train' % epoch)# 待编辑代码
    if tune_all:
        print("true")

    fea_hooks = get_feas_by_hook(net)  

    #net.train()
    net.eval()

    

    # get the watermark images
    wminputs, wmtargets = [], []

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        print('The number of hooks is:', len(fea_hooks))
        print('The shape of the first Conv2D feature is:', fea_hooks[0].fea.shape)
        #progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total


def ft_part(epoch, net, criterion, optimizer, logfile, loader, device, batch_train,tune_all=True):
    print('\nEpoch: %d' % epoch) #部分训练集用于微调
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []

        #B, C, H, W =inputs.size()   jianyan
        #print(batch_idx,B,targets[3])

    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx == batch_train.size:
            print(batch_idx)
            break
        if  batch_train[batch_idx] :
            iteration += 1
            #print(batch_idx)
            #print(batch_train[batch_idx])
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(loader), 'batch_idx:%d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx,train_loss / (iteration + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'% (train_loss / (iteration + 1), 100. * correct / total, correct, total))


def train_L2(epoch, net, teachernet, criterion, optimizer, logfile, loader, device, weight_reg_param, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    
    reg_loss = 0
    
    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        l2_loss = nn.MSELoss(reduction='sum')
        reg_loss = 0
        for param_id, (paramA, paramB) in enumerate(zip(net.parameters(), teachernet.parameters())):
            reg_loss += l2_loss(paramA, paramB)
        factor = weight_reg_param
        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss += reg_loss * factor

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))




def train_Logits(epoch, net, teachernet, criterion, optimizer, logfile, loader, device,  lmbda_d , wmloader=False, tune_all=True):
    print('train with logits')
    print('\nEpoch: %d' % epoch)
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    
    reg_loss = 0
    
    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)


        # add wmimages and targets
        if wmloader:
            inputsp = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targetsp = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputsp = net(inputsp)

        outputs = net(inputs)
        outputst = teachernet(inputs)

        loss = criterion(outputsp, targetsp)
        loss += lmbda_d * ((outputst - outputs) ** 2).mean()
        loss /= (lmbda_d + 1)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss(add wm with logit): %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def train_Logits_snnl(epoch, net, teachernet, criterion, optimizer, logfile, loader, device,  lmbda_d , temprature,snnl_weight,wmloader=False, tune_all=True):
    print('train with logits')
    print('\nEpoch: %d' % epoch)
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    
    reg_loss = 0
    
    fea_hooks = get_feas_by_hook(net)
    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        l2_loss = nn.MSELoss(reduction='sum')
        reg_loss = 0

        # add wmimages and targets
        if wmloader:
            inputsp = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targetsp = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)


        optimizer.zero_grad()
        outputsp = net(inputsp)

        losses_snnl=[]
        #add snnl loss
        for i, activation in enumerate(fea_hooks):#遍历feahook
            out_layer = activation.fea#
            print('batch',out_layer.shape[0])
            loss_snnl = snnl_loss(out_layer,targetsp,temprature)
            losses_snnl.append(loss_snnl)
       
        outputs = net(inputs)
        outputst = teachernet(inputs)

        

        loss = criterion(outputsp, targetsp)
        loss += lmbda_d * ((outputst - outputs) ** 2).mean()
        loss /= (lmbda_d + 1)
        print('loss with logit loss:',loss)
        print('snnl loss:',sum(losses_snnl)*snnl_weight)
        loss += sum(losses_snnl)*snnl_weight

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss',train_loss)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        #if batch_idx > 10:
        #    break

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss(add wm with logit): %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def train_snnl(epoch, net, criterion, optimizer, logfile, loader, device, temprature,snnl_weight,wmloader=False, tune_all=True):
    print('train with logits')
    print('\nEpoch: %d' % epoch)
    if tune_all:
        print("true")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    
    reg_loss = 0
    
    fea_hooks = get_feas_by_hook(net)
    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        reg_loss = 0

        # add wmimages and targets
        if wmloader:
            inputsp = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targetsp = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)


        optimizer.zero_grad()
        outputsp = net(inputsp)

        losses_snnl=[]
        #add snnl loss
        for i, activation in enumerate(fea_hooks):#遍历feahook
            out_layer = activation.fea#
            print('batch',out_layer.shape[0])
            loss_snnl = snnl_loss(out_layer,targetsp,temprature)
            losses_snnl.append(loss_snnl)
       
        outputs = net(inputs)

        loss = criterion(outputsp, targetsp)
        
        
        print('loss with logit loss:',loss)
        print('snnl loss:',sum(losses_snnl)*snnl_weight)
        loss += sum(losses_snnl)*snnl_weight

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss',train_loss)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        #if batch_idx > 10:
        #    break

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss(add wm with logit): %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def projection_lp_norm(cur_model, orig_model, model_eps,eps_in_percent,percentiles, device,lp_norm='inf', print_norm=False,
                       calc_diff_percentiles=True):
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    with torch.no_grad():
        l2_loss = nn.MSELoss(reduction='mean')
        param_diff = torch.empty(0, 1)
        param_diff = torch.flatten(param_diff)
        param_diff_norm = torch.empty(0, 1)
        param_diff_norm = torch.flatten(param_diff_norm)
        param_diff = param_diff.to(device)
        param_diff_norm = param_diff_norm.to(device)
        if lp_norm == 'inf':
            for i, (cur_param, orig_param) in enumerate(zip(cur_model.parameters(), orig_model.parameters())):
                if i<54:
                    if eps_in_percent:#eps约束权值改变百分比
                        cur_param.data = torch.where(cur_param < orig_param * (1.0 - model_eps / 100.0), orig_param * (1.0 - model_eps / 100.0), cur_param)
                        cur_param.data = torch.where(cur_param > orig_param * (1.0 + model_eps / 100.0), orig_param * (1.0 + model_eps / 100.0), cur_param)
                    else:
                        cur_param.data = orig_param.data - torch.clamp(orig_param.data - cur_param.data, -1 * model_eps, model_eps)#eps约束权值改变绝对值

                        #clamp(input, min, max, out=None) 将输入input张量每个元素的范围限制到区间 [min,max]
                    if calc_diff_percentiles:
                        layer_diff = torch.abs(torch.flatten(cur_param - orig_param))# (||,||,||,||,||)
                        layer_diff_norm = torch.div(layer_diff, torch.abs(torch.flatten(orig_param)))# (|ori-cur|/|ori|,||,||,||,||)
                        #   if(i==1):
                        param_diff = torch.cat([layer_diff, param_diff], dim=0)
                        param_diff_norm = torch.cat([layer_diff_norm, param_diff_norm], dim=0)
                  
                if (print_norm and i < 5):
                    print(cur_param.shape)
                    print(l2_loss(cur_param, orig_param))# MSE loss
                    print(torch.norm(cur_param - orig_param, p=float("inf")))# 范数Linf
                    print("")

        if model_eps == 0:
            for (n1, param1), (n2, param2) in zip(cur_model.named_parameters(), orig_model.named_parameters()):
                #     print(param1, param2)
                assert torch.all(param1.data == param2.data) == True
        if calc_diff_percentiles:
            for i in percentiles:
                diff_percents[i] = calculate_tensor_percentile(param_diff, i)
                norm_diff_percents[i] = calculate_tensor_percentile(param_diff_norm, i)
    return [diff_percents, norm_diff_percents]


def cal_model_param(cur_model, orig_model, percentiles, device):
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    with torch.no_grad():
        l2_loss = nn.MSELoss(reduction='mean')
        param_diff = torch.empty(0, 1)
        param_diff = torch.flatten(param_diff)
        param_diff_norm = torch.empty(0, 1)
        param_diff_norm = torch.flatten(param_diff_norm)
        param_diff = param_diff.to(device)
        param_diff_norm = param_diff_norm.to(device)
        for i, (cur_param, orig_param) in enumerate(zip(cur_model.parameters(), orig_model.parameters())):
            if i<54:
                layer_diff = torch.abs(torch.flatten(cur_param - orig_param))# (||,||,||,||,||)
                layer_diff_norm = torch.div(layer_diff, torch.abs(torch.flatten(orig_param)))# (|ori-cur|/|ori|,||,||,||,||)
                param_diff = torch.cat([layer_diff, param_diff], dim=0)
                param_diff_norm = torch.cat([layer_diff_norm, param_diff_norm], dim=0)
 
        for i in percentiles:
            diff_percents[i] = calculate_tensor_percentile(param_diff, i)
            norm_diff_percents[i] = calculate_tensor_percentile(param_diff_norm, i)
    return [diff_percents, norm_diff_percents]


def train_AWP(epoch, net,model_eps, teachernet,criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    eps_in_percent = 1 #相对eps约束权值改变
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    percentiles = ['0', '25', '50', '75', '90']
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    conv_param_names = []
    conv_params = []
    sum = 0
    l2_loss = nn.MSELoss( reduction='sum')

    
    for name, param in net.named_parameters():#统计conv层参数？
        if "conv" in name:
            conv_params += [param]
            conv_param_names += [name]

    reg_loss = 0
    num_paramsA = 0
    num_paramsB = 0
    for param_id, (paramA, paramB) in enumerate(zip(net.parameters(), teachernet.parameters())):#求teachrnet和net的绝对L2损失
        reg_loss += l2_loss(paramA, paramB)
        num_paramsA += np.prod(list(paramA.shape))
        num_paramsB += np.prod(list(paramB.shape))


    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):#每个batch一次权值更新
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)#常见的交叉熵损失

        loss.backward()
        optimizer.step()#更新参数

        #限制权值改变
        diff_percents, norm_diff_percents= projection_lp_norm(net, teachernet, model_eps, eps_in_percent,percentiles,device, lp_norm='inf', print_norm = False, calc_diff_percentiles = True)
        
        if batch_idx==3:
            print('diff_percents')    
            print(diff_percents)       
            print(norm_diff_percents)  

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    
    #######
    diff_tensor = torch.cat(
        [(param_1 - param_2).view(-1) for param_1, param_2 in zip(net.parameters(), teachernet.parameters())], dim=0)
    x = torch.cat([param_2.view(-1) for param_2 in teachernet.parameters()], dim=0)

    l2_norm_diff = float(torch.norm(diff_tensor))
    l2_norm_orig = float(torch.norm(x))

    linf_norm_diff = float(torch.norm(diff_tensor, float('inf')))
    linf_norm_orig = float(torch.norm(x, float('inf')))

    l1_norm_diff = float(torch.norm(diff_tensor, 1))
    l1_norm_orig = float(torch.norm(x, 1))
    result_dict = {'Loss': float(train_loss / (batch_idx + 1)), 'Acc': 100. * correct / total, 'Correct': correct,
                   'Total': total,
                   'Param_diff': reg_loss, 'Total_param': num_paramsA, 
                   'l2_norm_diff': l2_norm_diff, 'l2_norm_orig': l2_norm_orig,
                   'linf_norm_diff': linf_norm_diff, 'linf_norm_orig': linf_norm_orig, 'l1_norm_diff': l1_norm_diff,
                   'l1_norm_orig': l1_norm_orig}


    print(result_dict)

    
    
    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        f.write('paramdif: %f |Total_param: %f \nl2_norm_orig %f |l2_norm_diff: %f\n'
                % (reg_loss, num_paramsA, l2_norm_orig, l2_norm_diff))
    return result_dict


    
def train_look(epoch, net, teachernet, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    percentiles = ['0', '25', '50', '75', '90']
    diff_percents = dict.fromkeys(percentiles)
    norm_diff_percents = dict.fromkeys(percentiles)
    
    sum = 0
    l2_loss = nn.MSELoss(reduction='sum')


    reg_loss = 0
    num_paramsA = 0
    num_paramsB = 0
    for param_id, (paramA, paramB) in enumerate(zip(net.parameters(), teachernet.parameters())):#求teachrnet和net的绝对L2损失
        reg_loss += l2_loss(paramA, paramB)
        num_paramsA += np.prod(list(paramA.shape))
        num_paramsB += np.prod(list(paramB.shape))


    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):#每个batch一次权值更新
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)#常见的交叉熵损失

        loss.backward()
        optimizer.step()#更新参数



        if batch_idx == 3:
            pr_flag = 1
        else:
            pr_flag = 0

        for i, (cur_param, orig_param) in enumerate(zip(net.parameters(), teachernet.parameters())):
                if i < 54:
                    if (i == 1 and pr_flag == 1):
                        print("#######")
                        print(cur_param.data)
                        print(orig_param.data)          

        #限制权值改变
        if batch_idx==3:
            diff_percents, norm_diff_percents = cal_model_param(net, teachernet, percentiles, device)
            print('diff_percents')    
            print(diff_percents)       
            print(norm_diff_percents)  

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    
    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

# train function in a teacher-student fashion
def train_teacher(epoch, net, criterion, optimizer, use_cuda, logfile, loader, wmloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            if use_cuda:
                wminput, wmtarget = wminput.cuda(), wmtarget.cuda()
            wminputs.append(wminput)
            wmtargets.append(wmtarget)
        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if wmloader:
            # add wmimages and targets
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Epoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # return the acc.
    return 100. * correct / total



def testSR(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets, _, _ = poison_data_all(inputs, targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss, 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test Poison results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss , 100. * correct / total, correct, total))
    # return the acc.
    return 100. * correct / total

def testSR_ori(net, criterion, logfile, loader, sample_index,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets, _, _ = poison_data(inputs, targets,sample_index)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss, 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test Poison results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss , 100. * correct / total, correct, total))
    # return the acc.
    return 100. * correct / total

def test_logit(net, logfile, loader, device):
    print("Test Logits!!!! After softmax")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            targets_np = targets.detach().cpu().int().numpy()
            outputs_np = outputs.detach().cpu().numpy()
            logits = softmax(outputs_np)
            sorted_logits = sorted(logits, key=lambda x: x, reverse=True)
            sorted_id = sorted(range(len(logits)), key=lambda k: logits[k], reverse=True)

            print("Target Label: ", targets)
            print("Sorted label: ", sorted_id)
            # print("Predict Logits: \n", logits)
            print("Sorted Logits: ", sorted_logits)
            

            # f.write('True label: %.3f | Logits: %.3f\n'
            #         % (targets_np, logits))
            f.write('True label: %d\n '  % (targets_np))
            f.write('Sorted label: %s\n '  % (str(sorted_id)))
            # f.write('Logits: \n'  % (logits))
            # f.write('Sorted Logits: %s\n '  % (str(sorted_logits)))


def softmax(x):
    """ softmax function """
    
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    
    x -= np.max(x, axis = 1, keepdims = True) #为了稳定地计算softmax概率，减去每一行最大值
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    x = np.squeeze(x)
    return x
    
