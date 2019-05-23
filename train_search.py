
import argparse
import os
import sys
import torch
import math
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset
import utils
from model_search import Network

parser = argparse.ArgumentParser(description = 'model searcher.')
parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--m_lr',type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum',type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='alpha weight decay')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16,help='num of init channels')
parser.add_argument('--layers', type=int,default=8,help='total number of layers')
parser.add_argument('--model_path', type=str,default='saved_models', help='path to save the model')
parser.add_argument('--cutout', type=bool, default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--mode', type=int,default=3, help='the training mode')
parser.add_argument('--data',type=str,default='./data',help='the data folder')
parser.add_argument('--num_workers', type=int,default=6,help='the number worker.')
args = parser.parse_args()

'''
    args.mode: (weight indicates the convolutional parameter, while alpha encodes the network architecture)
            0, joint update alpha and weight
            1, update weight and alpha by coordinate descent
            2, update weight and alpha by coordinate descent, except that we use old weight to update alpha
            3, use hessian gradient to update weight and alpha
'''

class Trainer:
    def __init__(self):
        def _create_dataset():
            train_transform, valid_transform = utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            num_train = len(train_data)
            indices   = list(range(num_train))
            ratio = 1.0 if args.mode == 0 else args.train_portion
            split     = int(np.floor(ratio * num_train))

            self.loader_train = torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True,
                num_workers=args.num_workers
            )

            if args.mode != 0:
                self.loader_eval = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                    pin_memory=True, 
                    num_workers=args.num_workers
                )

            valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
            self.loader_test = torch.utils.data.DataLoader(
                valid_data,
                batch_size=args.batch_size,
                pin_memory=True,
                num_workers=args.num_workers
            )
        
        def _build_model():
            self.model = nn.DataParallel(Network(C=args.init_channels,num_classes=10, layer=args.layers)).cuda()
            self.loss_func = torch.nn.CrossEntropyLoss().cuda()
            if args.mode == 0:
                self.opt = torch.optim.SGD(
                    self.model.parameters(),      
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd
                )

                self.opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, 
                    float(args.epochs), 
                    eta_min=args.m_lr
                )
            elif args.mode in [1, 2, 3]:
                alpha_sign = 'w_alpha'
                self.w_opt = torch.optim.SGD(
                    [p[1] for p in self.model.named_parameters() if p[0].find(alpha_sign) <  0],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.wd
                )

                self.alpha_opt = torch.optim.Adam(
                    [p[1] for p in self.model.named_parameters() if p[0].find(alpha_sign) >= 0],
                    lr=args.arch_lr, 
                    betas=(0.5, 0.999),
                    weight_decay=args.arch_lr
                )

                self.opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.w_opt, 
                    float(args.epochs), 
                    eta_min=args.m_lr
                )
            else:
                print('invalid search mode.')
                sys.exit(0)
        utils.reproduceable(2019)
        _create_dataset()
        _build_model()
    
    def save_model(self):
        save_folder = './trained_model'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(self.model.state_dict(), os.path.join(save_folder, 'nas_arch.pkl'))

    def eval(self):
        self.model.eval()
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(self.loader_test):
            input, target = input.cuda(), target.cuda()
            logits = self.model(input)
            loss = self.loss_func(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            loss, prec1, prec5 = float(loss), float(prec1), float(prec5)
            n = input.size(0)
            objs.update(loss,  n)
            top1.update(prec1, n)
            top5.update(prec5, n)

        self.model.train()
        return top1.avg, top5.avg, objs.avg

    def train_0(self):
        sche, opt = self.opt_sche, self.opt
        pre_best_acc = 0
        for epoch in range(args.epochs):
            self.model.train()
            sche.step()
            objs = utils.AvgrageMeter()
            top1 = utils.AvgrageMeter()
            top5 = utils.AvgrageMeter()
            lr   = sche.get_lr()[0]
            print('eopch: {} lr: {}'.format(epoch, round(lr, 8)))
            
            for step, (input, target) in enumerate(self.loader_train):
                n = input.size(0)
                input, target = input.cuda(), target.cuda()
                logits = self.model(input)
                loss = self.loss_func(logits, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                opt.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                loss, prec1, prec5 = float(loss), float(prec1), float(prec5)
                objs.update(loss,  n)
                top1.update(prec1, n)
                top5.update(prec5, n)

                print('train => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(objs.avg, 6), round(top1.avg, 4), round(top5.avg, 4)))

            #evalute the model
            acc_tp1, acc_tp5, obj_avg = self.eval()
            if acc_tp1 > pre_best_acc:
                print('found better architecture with [acc: {}]'.format(round(acc_tp1, 4)))
                pre_best_acc = acc_tp1
                self.save_model()

            print('test  => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(obj_avg, 6), round(acc_tp1, 4), round(acc_tp5, 4)))

    def show_weight(self):
        # w_alpha_normal    
        # w_alpha_reduction 
        print(self.model.module.w_alpha_normal[0])


    def train_1(self):
        sche, w_opt, a_opt = self.opt_sche, self.w_opt, self.alpha_opt
        pre_best_acc = 0
        loader_eval = iter(self.loader_eval)
        for epoch in range(args.epochs):
            self.model.train()
            sche.step()
            train_objs = utils.AvgrageMeter()
            train_top1 = utils.AvgrageMeter()
            train_top5 = utils.AvgrageMeter()

            eval_objs  = utils.AvgrageMeter()
            eval_top1  = utils.AvgrageMeter()
            eval_top5  = utils.AvgrageMeter()
            lr   = sche.get_lr()[0]
            print('eopch: {} lr: {}'.format(epoch, round(lr, 8)))
            
            for step, (train_input, train_target) in enumerate(self.loader_train):
                n = train_input.size(0)
                train_input, train_target = train_input.cuda(), train_target.cuda()
                train_logits = self.model(train_input)
                train_loss   = self.loss_func(train_logits, train_target)
                w_opt.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip) #clip gradient
                w_opt.step()

                train_prec1, train_prec5 = utils.accuracy(train_logits, train_target, topk=(1, 5))
                train_loss, train_prec1, train_prec5 = float(train_loss), float(train_prec1), float(train_prec5)
                train_objs.update(train_loss,  n)
                train_top1.update(train_prec1, n)
                train_top5.update(train_prec5, n)
                print('train => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(train_objs.avg, 6), round(train_top1.avg, 4), round(train_top5.avg, 4)))

                try:
                    (valid_input, valid_target) = loader_eval.next()
                except:
                    loader_eval = iter(self.loader_eval)
                    (valid_input, valid_target) = loader_eval.next()

                n = valid_input.shape[0]
                valid_input, valid_target = valid_input.cuda(), valid_target.cuda()
                valid_logits = self.model(valid_input)
                valid_loss = self.loss_func(valid_logits, valid_target)
                
                a_opt.zero_grad()
                valid_loss.backward()
                a_opt.step()
    
                valid_prec1, valid_prec5 = utils.accuracy(valid_logits, valid_target, topk=(1, 5))
                valid_loss, valid_prec1, valid_prec5 = float(valid_loss), float(valid_prec1), float(valid_prec5)
                eval_objs.update(valid_loss,  n)
                eval_top1.update(valid_prec1, n)
                eval_top5.update(valid_prec5, n)
                print('eval  => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(eval_objs.avg, 6), round(eval_top1.avg, 4), round(eval_top5.avg, 4)))

            #evalute the model
            acc_tp1, acc_tp5, obj_avg = self.eval()
            if acc_tp1 > pre_best_acc:
                print('found better architecture with [acc: {}]'.format(round(acc_tp1, 4)))
                pre_best_acc = acc_tp1
                self.save_model()

    def train_2(self):
        sche, w_opt, a_opt = self.opt_sche, self.w_opt, self.alpha_opt
        pre_best_acc = 0
        loader_eval = iter(self.loader_eval)
        w_alpha = [p[1] for p in self.model.named_parameters() if p[0].find('w_alpha') >= 0]
        for epoch in range(args.epochs):
            self.model.train()
            sche.step()
            train_objs = utils.AvgrageMeter()
            train_top1 = utils.AvgrageMeter()
            train_top5 = utils.AvgrageMeter()

            eval_objs  = utils.AvgrageMeter()
            eval_top1  = utils.AvgrageMeter()
            eval_top5  = utils.AvgrageMeter()
            lr   = sche.get_lr()[0]
            print('eopch: {} lr: {}'.format(epoch, round(lr, 8)))
            
            for step, (train_input, train_target) in enumerate(self.loader_train):
                try:
                    (valid_input, valid_target) = loader_eval.next()
                except:
                    loader_eval = iter(self.loader_eval)
                    (valid_input, valid_target) = loader_eval.next()
                n = valid_input.shape[0]
                valid_input, valid_target = valid_input.cuda(), valid_target.cuda()
                valid_logits = self.model(valid_input)
                valid_loss = self.loss_func(valid_logits, valid_target)

                grad_alpha = [grad.detach().clone() for grad in torch.autograd.grad(valid_loss, w_alpha)]

                valid_prec1, valid_prec5 = utils.accuracy(valid_logits, valid_target, topk=(1, 5))
                valid_loss, valid_prec1, valid_prec5 = float(valid_loss), float(valid_prec1), float(valid_prec5)
                eval_objs.update(valid_loss,  n)
                eval_top1.update(valid_prec1, n)
                eval_top5.update(valid_prec5, n)
                print('eval  => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(eval_objs.avg, 6), round(eval_top1.avg, 4), round(eval_top5.avg, 4)))

                n = train_input.size(0)
                train_input, train_target = train_input.cuda(), train_target.cuda()
                train_logits = self.model(train_input)
                train_loss   = self.loss_func(train_logits, train_target)
                w_opt.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip) #clip gradient
                w_opt.step()
                train_prec1, train_prec5 = utils.accuracy(train_logits, train_target, topk=(1, 5))
                train_loss, train_prec1, train_prec5 = float(train_loss), float(train_prec1), float(train_prec5)
                train_objs.update(train_loss,  n)
                train_top1.update(train_prec1, n)
                train_top5.update(train_prec5, n)
                print('train => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(train_objs.avg, 6), round(train_top1.avg, 4), round(train_top5.avg, 4)))

                #update alpha
                for (p, v) in zip(w_alpha, grad_alpha):
                    p.grad.copy_(v)

                a_opt.step()

            #evalute the model
            acc_tp1, acc_tp5, obj_avg = self.eval()
            if acc_tp1 > pre_best_acc:
                print('found better architecture with [acc: {}]'.format(round(acc_tp1, 4)))
                pre_best_acc = acc_tp1
                self.save_model()
    
    def train_3(self):
        p_alpha = [p[1] for p in self.model.named_parameters() if p[0].find('w_alpha') >= 0]
        p_convs = [p[1] for p in self.model.named_parameters() if p[0].find('w_alpha') <  0]
        p_param = p_alpha + p_convs

        sche, w_opt, a_opt = self.opt_sche, self.w_opt, self.alpha_opt

        pre_best_acc = 0
        loader_eval = iter(self.loader_eval)

        def _get_weight_():
            return [p.data.clone() for p in p_convs]
        
        def _get_alpha_():
            return [p.data.clone() for p in p_alpha]

        def _set_alpha_(value):
            for (v, p) in zip(value, p_alpha):
                p.data.copy_(v)
        
        def _set_weight_(value):
            for (v, p) in zip(value, p_convs):
                p.data.copy_(v)

        def _set_alpha_grad_(grads):
            for (p, g) in zip(p_alpha, grads):
                try:
                    p.grad.copy_(g)
                except:
                    p.grad = g.clone()
        
        def _set_weight_grad_(grads):
            for (p, g) in zip(p_convs, grads):
                p.grad.copy_(g)

        for epoch in range(args.epochs):
            self.model.train()
            sche.step()
            train_objs = utils.AvgrageMeter()
            train_top1 = utils.AvgrageMeter()
            train_top5 = utils.AvgrageMeter()

            eval_objs  = utils.AvgrageMeter()
            eval_top1  = utils.AvgrageMeter()
            eval_top5  = utils.AvgrageMeter()
            lr   = sche.get_lr()[0]
            print('eopch: {} lr: {}'.format(epoch, round(lr, 8)))
            
            for step, (train_input, train_target) in enumerate(self.loader_train):
                try:
                    (valid_input, valid_target) = loader_eval.next()
                except:
                    loader_eval = iter(self.loader_eval)
                    (valid_input, valid_target) = loader_eval.next()

                valid_input, valid_target = valid_input.cuda(), valid_target.cuda()    
                train_input, train_target = train_input.cuda(), train_target.cuda()

                w   = _get_weight_()
                train_logits = self.model(train_input)
                train_loss = self.loss_func(train_logits, train_target)
                g_ = torch.autograd.grad(train_loss, p_convs)
                try:
                    g_w = [(g + args.momentum * w_opt.state[p]).data.clone() for (g, p) in zip(g_, p_convs)]
                except:
                    g_w = [g.data.clone() for g in g_]
                
                w_t = [v-g*args.lr for (v,g) in zip(w, g_w)]
                
                _set_weight_(w_t)
                valid_logits = self.model(valid_input)
                valid_loss = self.loss_func(valid_logits, valid_target)
                g_a_w = [g_.data.clone() for g_ in torch.autograd.grad(valid_loss, p_param)] #alpha + covns
                
                valid_prec1, valid_prec5 = utils.accuracy(valid_logits, valid_target, topk=(1, 5))
                valid_loss, valid_prec1, valid_prec5 = float(valid_loss), float(valid_prec1), float(valid_prec5)
                n = valid_target.shape[0]
                eval_objs.update(valid_loss,  n)
                eval_top1.update(valid_prec1, n)
                eval_top5.update(valid_prec5, n)
                print('eval  => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(eval_objs.avg, 6), round(eval_top1.avg, 4), round(eval_top5.avg, 4)))

                g_a_l = g_a_w[:2] #left term of graident of alpha
                g_w_t = g_a_w[2:]

                R = 0.01 / math.sqrt(sum((w_*w_).sum() for w_ in w_t))
                
                w_n = [ w_ - R * g_w_t_ for (w_, g_w_t_) in zip(w, g_w_t)]
                w_p = [ w_ + R * g_w_t_ for (w_, g_w_t_) in zip(w, g_w_t)]

                _set_weight_(w_n)
                train_logits = self.model(train_input)
                train_loss = self.loss_func(train_logits, train_target)
                g_a_n = [g_a.data.clone() for g_a in torch.autograd.grad(train_loss, p_alpha)]
                
                _set_weight_(w_p)
                train_logits = self.model(train_input)
                train_loss = self.loss_func(train_logits, train_target)
                g_a_p = [g_a.data.clone() for g_a in torch.autograd.grad(train_loss, p_alpha)]

                g_a_r = [ (gr-gl)/(2*R) for (gr, gl) in zip(g_a_p, g_a_n) ]

                g_a = [ gl - args.lr*gr for (gl, gr) in zip(g_a_l, g_a_r)]

                _set_alpha_grad_(g_a)
                a_opt.step()
                
                train_logits = self.model(train_input)
                train_loss   = self.loss_func(train_logits, train_target)
                w_opt.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip) #clip gradient
                w_opt.step()
                train_prec1, train_prec5 = utils.accuracy(train_logits, train_target, topk=(1, 5))
                train_loss, train_prec1, train_prec5 = float(train_loss), float(train_prec1), float(train_prec5)
                n = train_logits.shape[0]
                train_objs.update(train_loss,  n)
                train_top1.update(train_prec1, n)
                train_top5.update(train_prec5, n)
                print('train => [(epoch: {})(loss: {}), (prec1: {}), (prec5: {})]'.format(epoch, round(train_objs.avg, 6), round(train_top1.avg, 4), round(train_top5.avg, 4)))

            #evalute the model
            acc_tp1, acc_tp5, obj_avg = self.eval()
            if acc_tp1 > pre_best_acc:
                print('found better architecture with [acc: {}]'.format(round(acc_tp1, 4)))
                pre_best_acc = acc_tp1
                self.save_model()

    def train(self):
        if args.mode == 0:
            self.train_0()
        elif args.mode == 1:
            self.train_1()
        elif args.mode == 2:
            self.train_2()
        elif args.mode == 3:
            self.train_3()
        else:
            print('invalid mode.')
            sys.exit(0)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()