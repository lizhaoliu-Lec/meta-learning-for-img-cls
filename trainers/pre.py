""" Trainer for pretrain phase. """
import tqdm
import os.path as osp

from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import ensure_path, Timer
from dataloader.samplers import CategoriesSampler
from dataloader.dataset_loader import DatasetLoader as Dataset
from .utils import Averager, count_acc

from models import MetaTransformLearner

__all__ = ['PreTrainer']


class _BaseTrainer(object):
    # TODO, fulfill base trainer to be inherited by pre trainer and meta trainer
    def __init__(self, args, phase):
        # set args to self attr
        for k, v in vars(args).items():
            self.__setattr__(k, v)

        self.dataset_dir = osp.join(self.dataset_basedir, self.dataset_name)

        # set up gpu
        self.use_gpu = len(self.gpus) != '-1' and torch.cuda.is_available()

        # load training set
        self.train_set = Dataset('train', self.dataset_dir, train_aug=True)
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=self.batch_size_pre,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=self.use_gpu)

        # load validation set
        self.val_set = Dataset('val', self.dataset_dir)
        self.val_sampler = CategoriesSampler(label=self.val_set.labels,
                                             n_batch=600,
                                             n_cls=self.way_pre,
                                             n_per=self.shot_pre + self.query_pre)

        self.val_loader = DataLoader(dataset=self.val_set,
                                     batch_sampler=self.val_sampler,
                                     num_workers=8,
                                     pin_memory=self.use_gpu)


class PreTrainer(object):

    def __init__(self, args):

        # set args to self attr
        for k, v in vars(args).items():
            self.__setattr__(k, v)

        self.dataset_dir = osp.join(self.dataset_basedir, self.dataset_name)

        # set up gpu
        self.use_gpu = self.gpu != '-1'

        # load training set
        self.train_set = Dataset('train', self.dataset_dir, train_aug=True)
        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=self.batch_size_pre,
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=self.use_gpu)

        # load validation set
        self.val_set = Dataset('val', self.dataset_dir)
        self.val_sampler = CategoriesSampler(label=self.val_set.labels,
                                             n_batch=600,
                                             n_cls=self.way_pre,
                                             n_per=self.shot_pre + self.query_pre)

        self.val_loader = DataLoader(dataset=self.val_set,
                                     batch_sampler=self.val_sampler,
                                     num_workers=8,
                                     pin_memory=self.use_gpu)

        # set pretrain class number
        num_class_pretrain = self.train_set.num_class
        self.Learner = MetaTransformLearner(args=args, num_cls=num_class_pretrain)

        # set the pretrain log
        self.train_log = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'max_acc': self.Learner.checkpoints_pre['max_metric'],
            'max_acc_epoch': 0,
        }

        # set tensorboardX
        self.log_dir = '/'.join([self.log_dir, self.Learner.time_pre])
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # generate the label for eval
        label = torch.arange(end=self.way_pre).repeat(self.shot_pre + self.query_pre)
        if self.use_gpu:
            self.Learner = self.Learner.cuda()
            label = label.cuda()

        self.label_shot = label[:self.way_pre * self.shot_pre]  # for few train
        self.label_query = label[self.way_pre * self.shot_pre:]  # for eval

    def train(self):
        """The function for the pre-train phase."""

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = self.Learner.epoch * len(self.train_loader)
        # Set tensorboardX
        writer = SummaryWriter(log_dir=self.Learner.pretrain_save_dir)

        # Start pretrain
        for epoch in range(self.Learner.epoch, self.max_epoch_pre + 1):
            # update learning rate
            self.Learner.scheduler_pre.step()

            # set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # update global count number
                global_count = global_count + 1

                if self.use_gpu:
                    batch = [_.cuda() for _ in batch]

                data, label = batch
                # set to right phase
                self.Learner.set_phase('pretrain')
                # output logits for model
                logits = self.Learner(data)
                # calculate train loss
                loss = F.cross_entropy(logits, label)
                # calculate train accuracy
                acc = count_acc(logits, label)
                # write the tensorboardX records
                writer.add_scalar('train/loss', float(loss), global_count)
                writer.add_scalar('train/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.Learner.optimizer_pre.zero_grad()
                loss.backward()
                self.Learner.optimizer_pre.step()

            # update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.Learner.set_phase('preval')
            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            # run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                data, _ = batch
                if self.use_gpu:
                    data = data.cuda()
                p = self.shot_pre * self.way_pre
                data_shot, data_query = data[:p], data[p:]

                logits = self.Learner(data_shot, self.label_shot, data_query)
                loss = F.cross_entropy(logits, self.label_query)
                acc = count_acc(logits, self.label_query)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # write the tensorboardX records
            writer.add_scalar('val/loss', float(val_loss_averager), epoch)
            writer.add_scalar('val/acc', float(val_acc_averager), epoch)
            # print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))

            # update best saved model
            if val_acc_averager > self.train_log['max_acc']:
                self.train_log['max_acc'] = val_acc_averager
                self.train_log['max_acc_epoch'] = epoch
                self.Learner.save_pretrained_model(epoch=epoch, max_metric=val_acc_averager, is_best=True)

            # save model every 10 epochs
            if epoch % 10 == 0:
                self.Learner.save_pretrained_model(epoch=epoch, max_metric=val_acc_averager, is_best=False)

            # update the logs
            self.train_log['train_loss'].append(train_loss_averager)
            self.train_log['train_acc'].append(train_acc_averager)
            self.train_log['val_loss'].append(val_loss_averager)
            self.train_log['val_acc'].append(val_acc_averager)

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(self.train_log['max_acc_epoch'],
                                                                  self.train_log['max_acc']))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.max_epoch_pre)))
        writer.close()
