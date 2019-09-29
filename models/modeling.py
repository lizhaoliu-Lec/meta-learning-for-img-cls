import time
import json
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import ensure_path

from .mini_resnet import get_network
from .mini_resnet import get_network_outputs_dim
from .model_utils import WeightGenerator

__all__ = ['MetaTransformLearner']


class Classifier(nn.Module):
    """Fully connected that can dynamically set the weight and bias."""

    def __init__(self, hidden_dim, num_cls):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_cls = num_cls
        self.vars = nn.ParameterList()
        self.weight = nn.Parameter(torch.ones([self.num_cls, self.hidden_dim]))
        self.bias = nn.Parameter(torch.zeros(self.num_cls))
        torch.nn.init.kaiming_normal_(self.weight)
        self.vars.append(self.weight)
        self.vars.append(self.bias)

    def forward(self, input, new_vars=None):
        if new_vars is None:
            new_vars = self.vars
        weight = new_vars[0]
        bias = new_vars[1]
        out = F.linear(input, weight, bias)
        return out

    def parameters(self):
        return self.vars


class _FewShotLearner(nn.Module):
    def __init__(self, args, num_cls, hidden_dim, **kwargs):
        super(_FewShotLearner, self).__init__()
        self.allow_modes = ['pre', 'meta']
        # set args to self attr
        for k, v in vars(args).items():
            self.__setattr__(k, v)
        self.hyper_parameters = vars(args)

        assert self.mode in self.allow_modes, 'found unexpected mode `%s`' % self.mode

        # two tail to distinguish args
        self.pre_tail = '_pre'
        self.meta_tail = '_meta'

        if self.time_pre == '' and self.resume_pre:
            raise ValueError('try to resume the pre trained model, but time_pre not found in args.')
        if self.time_meta == '' and self.resume_meta:
            raise ValueError('try to resume the  meta train model, but time_meta not found in args.')

        if self.mode == 'meta' and self.time_pre == '':
            raise ValueError('to meta train the model, the pretrain time must be given to find the checkpoint.')

        self.time_format = '%Y-%m-%d...%H.%M.%S'
        self.time = time.strftime(self.time_format, time.localtime(time.time()))
        if self.time_pre == '':
            self.time_pre = self.time
        if self.time_meta == '':
            self.time_meta = self.time

        # set universal share attr
        self.dataset_dir = osp.join(self.dataset_basedir, self.dataset_name)
        self.pretrain_save_dir = '/'.join([
            self.save_dir, self.dataset_name, 'pre', self.basenet_name, self.time_pre
        ])
        self.meta_save_dir = '/'.join([
            self.save_dir, self.dataset_name, 'meta', self.basenet_name, self.time_meta
        ])
        self.hyper_parameters_path = '/'.join([
            self.save_dir, self.dataset_name, '%s', self.basenet_name, self.time
        ])

        self.checkpoints_pre = dict(max_metric=0.0)
        self.checkpoints_meta = dict(max_metric=0.0)

        self.basenet = get_network(self.basenet_name)(meta=True if self.mode == 'meta' else False)
        self.basenet_dim = get_network_outputs_dim(self.basenet_name)
        self.pre_classifier = nn.Sequential(nn.Linear(self.basenet_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, num_cls))
        self.meta_classifier = Classifier(hidden_dim=self.basenet_dim, num_cls=self.way_pre)

        self.optimizer_pre = torch.optim.SGD(
            params=self.parameters(),
            lr=self.lr_pre,
            momentum=self.momentum_pre,
            nesterov=True,
            weight_decay=self.weight_decay_pre)
        self.scheduler_pre = torch.optim.lr_scheduler.StepLR(self.optimizer_pre,
                                                             step_size=self.step_size_pre,
                                                             gamma=self.gamma_pre)

        self.optimizer_meta = torch.optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, self.basenet.parameters())}],
            lr=self.lr_meta,
            momentum=self.momentum_meta,
            nesterov=True,
            weight_decay=self.weight_decay_meta)
        self.scheduler_meta = torch.optim.lr_scheduler.StepLR(self.optimizer_meta,
                                                              step_size=self.step_size_meta,
                                                              gamma=self.gamma_meta)

        self.epoch = 1

        # save hyper parameters
        self.save_hyper_parameters()

    def save_hyper_parameters(self):
        save_dir = self.hyper_parameters_path % self.mode
        ensure_path(save_dir)
        save_path = osp.join(save_dir, 'hyper.json')
        json.dump(self.hyper_parameters,
                  open(save_path, 'w'), indent=2)

    def forward(self, *input):
        raise NotImplementedError

    def pretrain_forward(self, *input):
        raise NotImplementedError

    def meta_forward(self, *input):
        raise NotImplementedError

    def save_pretrained_model(self, epoch, max_metric, is_best=False):
        ensure_path(self.pretrain_save_dir)
        self.epoch = epoch
        self.checkpoints_pre = {
            'epoch': epoch,
            'max_metric': max_metric,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer_pre.state_dict(),
            'scheduler': self.scheduler_pre.state_dict(),
        }

        suffix = 'best' if is_best else str(epoch)
        save_path = self.pretrain_save_dir + '/model_%s.bin' % suffix
        print('Save pretrained model to [%s]' % save_path)
        torch.save(self.checkpoints_pre, save_path)

    def load_pretrained_model(self, load_best=False):
        load_path_pattern = self.pretrain_save_dir + '/model_%s.bin'
        if load_best:
            load_path = load_path_pattern % 'best'
        else:
            candidate_load_paths = os.listdir(self.pretrain_save_dir)
            epochs = [c.split('_')[-1].split('.')[0] for c in candidate_load_paths]
            epochs = [e for e in epochs if e.isdigit()]
            epochs = sorted(epochs)
            latest_epoch = epochs[-1]
            load_path = load_path_pattern % latest_epoch

        print('Load pretrained model from [%s]' % load_path)
        self.checkpoints_pre = torch.load(load_path, map_location=lambda storage, loc: storage)
        checkpoints = self.checkpoints_pre
        print('Load epoch [%d], max_metric [%s]' % (checkpoints['epoch'], checkpoints['max_metric']))
        self.epoch = checkpoints['epoch']
        self.load_state_dict(checkpoints['state_dict'])
        # don't load optimizer, will be bug
        # self.optimizer_pre.load_state_dict(checkpoints['optimizer'])
        self.scheduler_pre.load_state_dict(checkpoints['scheduler'])

    def save_meta_model(self, epoch, max_metric, is_best=False):
        ensure_path(self.meta_save_dir)
        self.epoch = epoch
        self.checkpoints_meta = {
            'epoch': epoch,
            'max_metric': max_metric,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer_meta.state_dict(),
            'scheduler': self.scheduler_meta.state_dict(),
        }

        suffix = 'best' if is_best else str(epoch)
        save_path = self.meta_save_dir + '/model_%s.bin' % suffix
        print('Save pretrained model to [%s]' % save_path)
        torch.save(self.checkpoints_meta, save_path)

    def load_meta_model(self, load_best=False):
        load_path_pattern = self.meta_save_dir + '/model_%s.bin'
        if load_best:
            load_path = load_path_pattern % 'best'
        else:
            candidate_load_paths = os.listdir(self.meta_save_dir)
            epochs = [c.split('_')[-1].split('.')[0] for c in candidate_load_paths]
            epochs = [e for e in epochs if e.isdigit()]
            epochs = sorted(epochs)
            latest_epoch = epochs[-1]
            load_path = load_path_pattern % latest_epoch

        print('Load meta trained model from [%s]' % load_path)
        self.checkpoints_pre = torch.load(load_path)
        checkpoints = self.checkpoints_pre
        print('Load epoch [%d], max_metric [%s]' % (checkpoints['epoch'], checkpoints['max_metric']))
        self.epoch = checkpoints['epoch']
        self.load_state_dict(checkpoints['state_dict'])
        # self.optimizer_pre.load_state_dict(checkpoints['optimizer'])
        self.scheduler_pre.load_state_dict(checkpoints['scheduler'])


class MetaTransformLearner(_FewShotLearner):

    def __init__(self, args, num_cls=64):
        super(MetaTransformLearner, self).__init__(args, hidden_dim=1000, num_cls=num_cls)
        self.allow_phases = ['pretrain', 'meta_train', 'preval', 'meta_eval']
        assert self.phase in self.allow_phases, 'found unexpected phase `%s`' % self.phase

        # 2 cases
        # we need to load the best pretrain model first
        # (1) first time we meta train the model
        # (2) preval the model
        if (self.phase == 'meta_train' and not self.resume_meta) or self.phase == 'preval':
            self.load_pretrained_model(load_best=True)

        # 1 cases
        # we need to load the latest pretrain model first
        # (1) resume to pretrain the model
        if self.phase == 'pretrain' and self.resume_pre:
            self.load_pretrained_model(load_best=False)

        # 1 cases
        # we need to load the latest meta-train model first
        # (1) resume to meta-train the model
        if self.phase == 'meta-train' and self.resume_meta:
            self.load_pretrained_model(load_best=False)

        # 1 cases
        # we need to load the best meta train model
        # (1) meta-eval the model
        if self.phase == 'meta_eval':
            self.load_meta_model(load_best=True)

    def forward(self, *input):
        if self.phase == 'pretrain':
            return self.pretrain_forward(*input)
        if self.phase == 'preval':
            return self.meta_forward(*input)
        if self.phase == 'meta_train':
            return self.meta_forward(input)
        if self.phase == 'meta_eval':
            return self.meta_forward(input)

    def set_phase(self, phase):
        self.allow_phases = ['pretrain', 'preval', 'meta_train', 'meta_eval']
        assert phase in self.allow_phases, 'found unexpected phase `%s`' % phase
        self.phase = phase
        if self.phase == 'pretrain' or self.phase == 'meta_train':
            self.train()
        else:
            self.eval()

    def pretrain_forward(self, input):
        return self.pre_classifier(self.basenet(input))

    def meta_forward(self, *input):
        data_shot, label_shot, data_query = input
        embedding_query = self.basenet(data_query)
        embedding_shot = self.basenet(data_shot)
        new_weights = self.meta_classifier.parameters()

        fast_step = self.fast_step_pre if self.mode == 'pre' else self.fast_step_meta
        fast_lr = self.fast_lr_pre if self.mode == 'pre' else self.fast_lr_meta
        for _ in range(fast_step):
            logits_shot = self.meta_classifier(embedding_shot, new_weights)
            loss_shot = F.cross_entropy(logits_shot, label_shot)
            grad = torch.autograd.grad(loss_shot, new_weights)
            new_weights = list(map(lambda p: p[1] - fast_lr * p[0], zip(grad, new_weights)))
        logits_query = self.meta_classifier(embedding_query, new_weights)
        return logits_query


class WithoutForgettingLearner(_FewShotLearner):
    def __init__(self, args, num_cls=64, hidden_dim=1000):
        super(WithoutForgettingLearner, self).__init__(args, hidden_dim=hidden_dim, num_cls=num_cls)
        self.allow_phases = ['pretrain', 'meta_train', 'preval', 'meta_eval']
        assert self.phase in self.allow_phases, 'found unexpected phase `%s`' % self.phase

        self.hidden_dim = hidden_dim
        # weight generator instance
        self.weight_generator = WeightGenerator(base_cls=num_cls,
                                                # FIXME: There is `self.way_pre` above,
                                                #  but seems like it is undeclared
                                                way=self.way_pre,
                                                hidden_dim=hidden_dim)

        # 2 cases
        # we need to load the best pretrain model first
        # (1) first time we meta train the model
        # (2) preval the model
        if (self.phase == 'meta_train' and not self.resume_meta) or self.phase == 'preval':
            self.load_pretrained_model(load_best=True)

        # 1 cases
        # we need to load the latest pretrain model first
        # (1) resume to pretrain the model
        if self.phase == 'pretrain' and self.resume_pre:
            self.load_pretrained_model(load_best=False)

        # 1 cases
        # we need to load the latest meta-train model first
        # (1) resume to meta-train the model
        if self.phase == 'meta-train' and self.resume_meta:
            self.load_pretrained_model(load_best=False)

        # 1 cases
        # we need to load the best meta train model
        # (1) meta-eval the model
        if self.phase == 'meta_eval':
            self.load_meta_model(load_best=True)

    def forward(self, *input):
        if self.phase == 'pretrain':
            return self.pretrain_forward(*input)
        if self.phase == 'preval':
            return self.meta_forward(*input)
        if self.phase == 'meta_train':
            return self.meta_forward(input)
        if self.phase == 'meta_eval':
            return self.meta_forward(input)

    def set_phase(self, phase):
        self.allow_phases = ['pretrain', 'preval', 'meta_train', 'meta_eval']
        assert phase in self.allow_phases, 'found unexpected phase `%s`' % phase
        self.phase = phase
        if self.phase == 'pretrain' or self.phase == 'meta_train':
            self.train()
        else:
            self.eval()

    def pretrain_forward(self, input):
        return self.pre_classifier(self.basenet(input))

    def meta_forward(self, *input):
        data_shot, label_shot, data_query = input
        embedding_query = self.basenet(data_query)
        embedding_shot = self.basenet(data_shot)

        base_weight = self.pre_classifier[-1].weight.data
        novel_weight = self.weight_generator.get_novel_weight(embedding_shot, base_weight)
        self.meta_classifier.weight.data = novel_weight
        logits_query = self.meta_classifier(embedding_query)
        return logits_query
