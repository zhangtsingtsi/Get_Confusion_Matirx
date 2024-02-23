#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from utils_dir.utils_result import get_result_confusion_jsons

# from torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import logging
import pickle

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def setup_logger(name, log_file, level = logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/ntu60_xsub_body/',
        help='the work folder for storing results')
    parser.add_argument(
        '--logger-dir',
        default='./runs_psumnet_ntu60_xsub_body/',
        help='the logging folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/ntu_config/body.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=3407, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=0,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=24,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        # action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer') # Adam SGD
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    print('Dir not removed: ', arg.model_saved_name)
                    # answer = input('delete it? y/n:')
                    # if answer == 'y':
                    #     shutil.rmtree(arg.model_saved_name)
                    #     print('Dir removed: ', arg.model_saved_name)
                    #     input('Refresh the website of tensorboard by pressing any keys=')
                    # else:
                    #     print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(self.arg.logger_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(self.arg.logger_dir, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.logger_dir, 'test'), 'test')
        # logger for training accuracies
        self.train_logger = setup_logger('Training_accuracy', os.path.join(self.arg.logger_dir, 'train_output.log'))

        # logger for evaluation accuracies
        self.eval_logger = setup_logger('Evaluation_accuracy', os.path.join(self.arg.logger_dir,'eval_output.log'))

        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = 0
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                              patience=5, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

        # 从保存的状态中恢复优化器，则需要显式地在参数组中设置 initial_lr
        # for param_group in self.optimizer.param_groups:
        #     param_group.setdefault('initial_lr', param_group['lr'])
            
        # self.lr_scheduler = MultiStepLR(self.optimizer, self.arg.step, gamma=0.5, last_epoch=self.arg.start_epoch)

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        if not os.path.exists(self.arg.logger_dir):
            os.makedirs(self.arg.logger_dir)

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.weight_decay ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=True):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        # self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        use_amp = True
        # 用于自动化混合精度训练中的梯度缩放。
        # 混合精度训练是一种优化技术，它通过在模型训练过程中同时使用单精度（float32）和半精度（float16）
        # 来提高训练速度和减少内存占用，同时尽量保持训练质量。
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # 1.初始化一个GradScaler对象

        for batch_idx, (x, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                x = x.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            #output = self.model(x)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output, z = self.model(x, F.one_hot(label, num_classes=self.model.num_class))
                loss = self.loss(output, label)

            # backward 在训练阶段step优化器，在验证阶段step学习率调整策略lr_scheduler
            self.optimizer.zero_grad()
            #loss.backward()
            #self.optimizer.step()
            scaler.scale(loss).backward() # 2. 使用scaler.scale(loss)缩放损失，并反向传播
            scaler.step(self.optimizer) # 3. 更新模型权重
            scaler.update() # 4. 更新缩放因子
            

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # self.lr_scheduler.step() # MultiStepLR

        # statistics of time consumption and loss
        self.train_writer.add_scalar('acc', np.mean(acc_value)*100, epoch + 1)
        self.train_writer.add_scalar('loss', np.mean(loss_value), epoch + 1)

        self.train_logger.info("Average training loss after {0} epochs is {1}".format(epoch + 1, np.mean(loss_value)))
        self.train_logger.info("Average training accuracy after {0} epochs is {1}".format(epoch + 1, np.mean(acc_value)*100))

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  LR: {}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), self.lr, np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=True, loader_name=['test'], wrong_file=None, result_file=None):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (x, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    x = x.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    #output = self.model(x)
                    output, z = self.model(x, F.one_hot(label, num_classes=self.model.num_class))
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            # ReduceLROnPlateau 根据模型在验证集上的性能来调整学习率
            # 因此，它通常在验证阶段（val）之后使用，并且需要传递一个性能指标作为参数（通常是验证集的损失）
            self.lr_scheduler.step(loss) # ReduceLROnPlateau

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, epoch + 1)
                self.val_writer.add_scalar('acc', 100*accuracy, epoch + 1)
                self.eval_logger.info("Average testing loss after {0} epochs is {1}".format(epoch + 1, loss))
                self.eval_logger.info("Average testing accuracy after {0} epochs is {1}".format(epoch + 1, 100*accuracy))


            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
                
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f) # 保存评分，用于后续的分析和比较
            
            ####### get confusion matrix
            # 收集所有batch的预测和真实标签
            all_pred_labels = np.concatenate(pred_list, axis=0)
            all_true_labels = np.concatenate([lab.numpy() for lab in label_list], axis=0)
            # 调用函数计算混淆矩阵和每个类别的精度
            acc_dir = os.path.join(self.arg.work_dir, 'accuracy_info')
            if not os.path.exists(acc_dir):
                os.makedirs(acc_dir)
            acc_f_name_prefix = '{}/epoch_{}_{}'.format(acc_dir, epoch + 1, ln)
            get_result_confusion_jsons(all_true_labels, all_pred_labels, self.arg.train_feeder_args['data_path'], acc_f_name_prefix)
            #######

            ###### 计算混淆矩阵
            cm = confusion_matrix(all_true_labels, all_pred_labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
            con_max_dir = os.path.join(self.arg.work_dir, 'confusion_matrix')
            if not os.path.exists(con_max_dir):
                os.makedirs(con_max_dir)
            # 绘制并保存热力图
            plt.figure(figsize=(10, 10))
            sns.heatmap(cm_normalized, annot=False, cmap='Blues')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title(f'Confusion Matrix') # Confusion Matrix at Epoch {epoch + 1}
            plt.savefig(os.path.join(con_max_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
            plt.close()
            ######

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

if __name__ == '__main__':
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys() #vars(p)获取对象p的属性字典
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
