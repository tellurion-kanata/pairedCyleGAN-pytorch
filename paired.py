import torch
import torch.nn as nn
import torch.optim as optim
import data.utils as utils
import itertools

from models import *


class DraftDrawer(BaseModel):
    eps = 1e-7

    def __init__(self, opt):
        super(DraftDrawer, self).__init__(opt)
        self.initialize()


    def initialize(self):
        self.opt_model = 'draftdrawer'

        # Reference encoder is fixed.
        self.G = init_net(DeepResidualNetwork(1, 3), gpus=self.gpus)

        if not self.opt.eval:
            self.F = init_net(DeepResidualNetwork(3, 1, True), gpus=self.gpus)
            self.Dg = define_D(
                input_channels = self.opt.chA,
                n_layers = self.opt.n_layer,
                ndf = self.opt.ndf,
                spec_norm = self.opt.use_spec,
                gpus = self.gpus
            )
            self.Df = define_D(
                input_channels = self.opt.chB,
                n_layers = self.opt.n_layer,
                ndf = self.opt.ndf,
                spec_norm = self.opt.use_spec,
                gpus = self.gpus
            )

            self.optimizer_G = optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()), lr=self.lr, betas=self.betas)
            self.optimizer_D = optim.Adam(itertools.chain(self.Dg.parameters(), self.Df.parameters()), lr=self.lr, betas=self.betas)

            self.criterion_GAN = loss.GANLoss(self.opt.gan_mode).to(self.device)
            self.criterion_L1 = nn.L1Loss().to(self.device)

            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.models = {'G': self.G, 'F': self.F, 'Dg': self.Dg, 'Df': self.Df}
        else:
            self.models = {'G': self.G}
        self.setup()


    def read_input(self, input):
        self.real_A = input['sketch'].to(self.device)
        self.real_B = input['color'].to(self.device)
        self.x_idx = input['index']

    def forward(self):
        self.fake_B = self.G(self.real_A, self.real_B)
        self.fake_A = self.F(self.real_B)
        self.rec_B = self.G(self.fake_A, self.fake_B)
        self.rec_A = self.F(self.fake_B)

    def train(self):
        self.step = self.opt.start_step
        if self.opt.resume:
            self.load(self.opt.load_epoch, resume=True)

        for epoch in range(self.st_epoch, self.ed_epoch):
            for idx, data in enumerate(self.datasets):
                self.read_input(data)
                self.forward()

                # update discriminator
                self.set_requires_grad([self.Dg, self.Df], True)
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                # update draft generator
                self.set_requires_grad([self.Dg, self.Df], False)
                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()

                if idx % self.save_freq_step == 0:
                    self.output_samples(epoch, idx)
                    self.save()
                if idx % self.print_state_freq == 0:
                    self.set_state_dict()
                    self.print_training_iter(epoch, idx)

                self.step += 1

            self.save()
            self.set_state_dict()
            self.print_training_epoch(epoch)

            for scheduler in self.schedulers:
                scheduler.step()

            if epoch % self.save_freq == 0:
                self.save(epoch)

    def basic_backward_D(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterion_GAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_Dg = self.basic_backward_D(self.Dg, self.real_B, self.fake_B)
        self.loss_Df = self.basic_backward_D(self.Df, self.real_A, self.fake_A)


    def backward_G(self):
        self.loss_G = self.criterion_GAN(self.Dg(self.fake_B), True)
        self.loss_F = self.criterion_GAN(self.Df(self.fake_A), True)

        self.loss_identity = self.criterion_L1(self.F(self.fake_B), self.real_A)
        self.loss_style = self.criterion_L1(self.G(self.fake_A, self.fake_B), self.real_B)

        self.loss_G = self.loss_F + self.loss_G + self.loss_style + self.loss_identity
        self.loss_G.backward()


    def set_state_dict(self):
        self.state_dict = {
            'loss_Dg': self.loss_Dg,
            'loss_Df': self.loss_Df,
            'loss_G': self.loss_G,
            'loss_F': self.loss_F,
            'loss_id': self.loss_identity,
            'loss_style': self.loss_style
        }