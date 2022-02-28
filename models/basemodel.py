import shutil

import torch
import data.utils as utils

from torch.utils.tensorboard import SummaryWriter
from datasets.datasets import CustomDataLoader
from models.loss import get_scheduler

import os
import time

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.ckpt_path = os.path.join(opt.save_path,opt.name)
        self.sample_path = os.path.join(self.ckpt_path, 'sample')
        self.test_path = os.path.join(self.ckpt_path, 'test')

        if not opt.eval:
            self.lr = opt.learning_rate
            self.betas = (opt.beta1, opt.beta2)
            self.save_freq = opt.save_freq
            self.save_freq_step = opt.save_freq_step
            self.print_state_freq = opt.print_state_freq
            self.ed_epoch = opt.niter + opt.niter_decay + 1
            self.st_epoch = opt.epoch_count

        self.gpus = opt.gpus
        self.device = torch.device('cuda:{}'.format(self.gpus[0])) if self.gpus else torch.device('cpu')


    def setup(self):
        if not self.opt.eval:
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        # torch.backends.cudnn.benchmark = True
        self.batch_size = self.opt.batch_size
        self.data_loader = CustomDataLoader()
        self.data_loader.initialize(self.opt)
        self.datasets = self.data_loader.load_data()
        self.data_size = len(self.datasets)

        if not self.opt.eval:
            self.print_param()
            self.writer = SummaryWriter(os.path.join(self.ckpt_path, 'logs'))

        self.start_time = time.time()
        self.pre_epoch_time = self.start_time
        self.pre_iter_time = self.start_time


    def eval(self):
        for net in self.models.keys():
            self.models[net].eval()


    def print_param(self):
        arch_log = open(os.path.join(self.ckpt_path, 'nets_arch.txt'), 'wt')
        for net in self.models.keys():
            arch_log.write('{:>40}:\n'.format('Network ' + net))
            for name, param in self.models[net].named_parameters():
                arch_log.write('{}: {}\n'.format(name, param.size()))
            arch_log.write('\n')


    def save(self, epoch='latest'):
        if epoch != 'latest':
            training_state = {'epoch': epoch, 'lr': self.lr, 'iterations': self.step}
            torch.save(training_state, os.path.join(self.ckpt_path, 'model_states.pth'))

        for net in self.models.keys():
            torch.save(self.models[net].state_dict(), os.path.join(self.ckpt_path, '{}_{}_params.pth'.format(epoch, net)))


    def load(self, epoch='latest', resume=False):
        print('\n**************** loading model ******************')

        for net in self.models.keys():
            file_path = os.path.join(self.ckpt_path, '{}_{}_params.pth'.format(epoch, net))
            if not os.path.exists(file_path):
                raise FileNotFoundError('%s is not found.' % file_path)
            state_dict = torch.load(file_path)
            required_state_dict = self.models[net].state_dict()
            for key in required_state_dict.keys():
                load_key = key
                if self.device == torch.device('cpu'):
                    load_key = 'module.' + load_key
                assert load_key in state_dict.keys()
                required_state_dict[key] = state_dict[load_key]
            self.models[net].load_state_dict(required_state_dict)
            print('\n********** load model [{}] successfully **************'.format(net))

        if resume:
            states = torch.load(os.path.join(self.ckpt_path, 'model_states.pth'))
            try:
                self.lr = states['lr']
                self.step = states['iterations']
            except:
                print('Training states file error.')


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']


    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    # output training message iter ver.
    def print_training_iter(self, epoch, idx):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        self.pre_iter_time = current_time
        message = 'iter_time: %4.4f s, epoch: [%d/%d], step: [%d/%d], global_step: [%d], learning_rate: %.7f' % \
                  (iter_time, epoch, self.ed_epoch-1, idx+1, self.data_size // self.batch_size, self.step, self.lr)

        print(message, end='')
        train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
        train_log.write(message)

        for label in self.state_dict.keys():
            print(', %s: %.7f' % (label, self.state_dict[label]), end='')
            train_log.write(', %s: %.7f' % (label, self.state_dict[label]))
            self.writer.add_scalar(label, self.state_dict[label], global_step=self.step)

        print('')
        train_log.write('\n')
        train_log.close()


    # output training message epoch ver.
    def print_training_epoch(self, epoch):
        current_time = time.time()
        epoch_time = current_time - self.pre_epoch_time
        total_time = current_time - self.start_time
        self.pre_epoch_time = current_time
        message = 'total time: %4.4f s, epoch_time: %4.4f s, epoch: [%d/%d], learning_rate: %.7f' % \
                  (total_time, epoch_time, epoch, self.ed_epoch-1, self.lr)

        print(message, end='')
        train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
        train_log.write(message)

        for label in self.state_dict.keys():
            print(', %s: %.7f' % (label, self.state_dict[label]), end='')
            train_log.write(', %s: %.7f' % (label, self.state_dict[label]))
        print('')
        train_log.write('\n')
        train_log.close()


    @torch.no_grad()
    def output_samples(self, epoch, index):
        self.G.eval()
        self.F.eval()
        self.forward()

        utils.save_image(self.real_A[0].cpu(), os.path.join(self.sample_path, '{}_{}_realA.png'.format(epoch, index)), grayscale=True)
        utils.save_image(self.real_B[0].cpu(), os.path.join(self.sample_path, '{}_{}_realB.png'.format(epoch, index)))
        utils.save_image(self.fake_A[0].cpu(), os.path.join(self.sample_path, '{}_{}_fakeA.png'.format(epoch, index)), grayscale=True)
        utils.save_image(self.fake_B[0].cpu(), os.path.join(self.sample_path, '{}_{}_fakeB.png'.format(epoch, index)))
        utils.save_image(self.rec_A[0].cpu(), os.path.join(self.sample_path, '{}_{}_recA.png'.format(epoch, index)), grayscale=True)
        utils.save_image(self.rec_B[0].cpu(), os.path.join(self.sample_path, '{}_{}_recB.png'.format(epoch, index)))

        self.G.train()
        self.F.train()


    def train(self):
        pass

    def test(self, epoch='latest'):
        def mk_dirs(path):
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                shutil.rmtree(path)
                os.makedirs(path)

        size_flag = '_resize' if self.opt.resize else ''
        epoch_flag = '_latest' if epoch == 'latest' else '_epoch' + epoch
        filename = os.path.basename(self.opt.dataroot) + epoch_flag + size_flag              # replace '/' and '\' with other file split char in your system

        test_path = os.path.join(self.test_path, filename)

        with torch.no_grad():
            self.load(epoch)
            self.generator.eval()
            test_size = len(self.datasets)

            for idx, data in enumerate(self.datasets):
                self.read_input(data)
                self.forward()

                index = self.x_idx[0]
                utils.save_image(self.real_A.squeeze(0).cpu(),
                                 os.path.join(test_path, '{}_realA.png'.format(idx)), grayscale=True)
                utils.save_image(self.real_B.squeeze(0).cpu(),
                                 os.path.join(test_path, '{}_realB.png'.format(idx)))
                utils.save_image(self.fake_A.squeeze(0).cpu(),
                                 os.path.join(test_path, '{}_fakeA.png'.format(idx)), grayscale=True)
                utils.save_image(self.fake_B.squeeze(0).cpu(),
                                 os.path.join(test_path, '{}_fakeB.png'.format(idx)))
                print('test proces: [{} / {}] ...'.format(idx + 1, test_size))
                if idx == test_size - 1:
                    break


    def initialize(self):
        pass

    def forward(self):
        pass

    def read_input(self, input):
        pass

    def set_state_dict(self):
        pass

    def backward_D(self):
        pass

    def backward_G(self):
        pass
