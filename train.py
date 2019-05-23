
import argparse
import os
import sys
import torch
import math
import numpy as np
import torch.nn as nn
import torchvision.datasets as dset
import utils
from model import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', type=bool, default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', type=bool, default=True, help='use cutout')
parser.add_argument('--num_workers', type=int, default=6,help='the number of workers.')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

args = parser.parse_args()

class Trainer:
    def __init__(self):
        def _create_dataset():
            train_transform, valid_transform = utils._data_transforms_cifar10(args)

            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            self.loader_train = torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch_size,
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
            pth = torch.load(os.path.join('./trained_model', 'nas_arch.pkl'))
            arch = {
                'normal' : pth['module.w_alpha_normal'].cpu().numpy(),
                'reduce' : pth['module.w_alpha_reduction'].cpu().numpy()
            }
            self.model = nn.DataParallel(NetworkCIFAR(C=args.init_channels,num_classes=10, layers=args.layers,drop_path_prob=args.drop_path_prob, arch=arch)).cuda()
            self.loss_func = torch.nn.CrossEntropyLoss().cuda()

            self.opt = torch.optim.SGD(
                self.model.parameters(),      
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.wd
            )

            self.opt_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, 
                float(args.epochs)
            )

        utils.reproduceable(2019)
        _create_dataset()
        _build_model()
    
    def save_model(self):
        save_folder = './trained_model'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(self.model.state_dict(), os.path.join(save_folder, 'nas_model.pkl'))

    def eval(self):
        self.model.eval()
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(self.loader_test):
            input, target = input.cuda(), target.cuda()
            logits, _ = self.model(input)
            loss = self.loss_func(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            loss, prec1, prec5 = float(loss), float(prec1), float(prec5)
            n = input.size(0)
            objs.update(loss,  n)
            top1.update(prec1, n)
            top5.update(prec5, n)

        self.model.train()
        return top1.avg, top5.avg, objs.avg

    def train(self):
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

            self.model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            for step, (input, target) in enumerate(self.loader_train):
                n = input.size(0)
                input, target = input.cuda(), target.cuda()
                logits, logits_aux = self.model(input)
                loss = self.loss_func(logits, target)
                if args.auxiliary:
                    aux_loss = self.loss_func(logits_aux, target)
                    loss += aux_loss * args.auxiliary_weight
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                opt.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                loss, prec1, prec5 = float(loss), float(prec1), float(prec5)
                objs.update(loss,  n)
                top1.update(prec1, n)
                top5.update(prec5, n)

                print('train => [(epoch: {}), (loss: {}), (prec1: {}), (prec5: {})]'.format(epoch, round(objs.avg, 6), round(top1.avg, 4), round(top5.avg, 4)))

            #evalute the model
            acc_tp1, acc_tp5, obj_avg = self.eval()
            if acc_tp1 > pre_best_acc:
                print('found better architecture with [acc: {}]'.format(round(acc_tp1, 4)))
                pre_best_acc = acc_tp1
                self.save_model()

            print('test  => [(loss: {}), (prec1: {}), (prec5: {})]'.format(round(obj_avg, 6), round(acc_tp1, 4), round(acc_tp5, 4)))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()