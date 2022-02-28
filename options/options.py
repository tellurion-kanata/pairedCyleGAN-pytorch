import os
import argparse
import torch

class Options():
    def initialize(self, eval=False):
        self.eval = eval

        parser = argparse.ArgumentParser()
        # overall options
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataroot', '-d', required=True,
                            help='Training dataset')
        parser.add_argument('--gpus', type=str, default='0',
                            help='gpu ids:  e.g. 0 | 0,1 | 0,2 | -1 for cpu')
        parser.add_argument('--batch_size', '-bs', default=16, type=int,
                            help='Number of batch size')
        parser.add_argument('--load_epoch', '-le', type=str, default='latest',
                            help='Epoch to load. Default is \'latest\'')
        parser.add_argument('--load_size', type=int, default=256,
                            help='Loaded size of image')
        parser.add_argument('--num_threads', '-nt', type=int, default=0,
                            help='Number of threads when reading data')
        parser.add_argument('--image_pattern', '-p', type=str, default='*.jpg',
                            help='pattern of training images')
        parser.add_argument('--save_path', '-s', type=str, default='./checkpoints',
                            help='Trained models save path')
        parser.add_argument('--no_shuffle', action='store_true',
                            help='Not to shuffle data every epoch')
        parser.add_argument('--chA', type=int, default=1,
                            help='Channels of real A image')
        parser.add_argument('--chB', type=int, default=3,
                            help='Channels of real B image')
        return parser

    def modify_options(self, parser):
        opt, _ = parser.parse_known_args()
        if not self.eval:
            parser = self.add_training_options(parser)
        else:
            parser.add_argument('--resize', action='store_true',
                                help='Resize image to load size during evaluation')
        return parser


    def add_training_options(self, parser):
        parser.add_argument('--resume', action='store_true',
                            help='Resume training')

        # training options
        parser.add_argument('--niter', type=int, default=10,
                            help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=10,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                            help='Initial earning rate')
        parser.add_argument('--gan_mode', type=str, default='vanilla',
                            help='Type of GAN loss function [vanilla (default) | lsgan | wgangp]')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='Policy of learning rate decay')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='Beta1 for Adam')
        parser.add_argument('--beta2', type=float, default=0.99,
                            help='Beta2 for Adam')
        parser.add_argument('--start_step', type=int, default=0,
                            help='Start step')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='Start epoch')

        # image pre-processing and training states output
        parser.add_argument('--image_size', type=int, default=512,
                            help='Original size of training image')
        parser.add_argument('--no_flip', action='store_true',
                            help='Not to flip image')
        parser.add_argument('--no_crop', action='store_true',
                            help='Not to crop image')
        parser.add_argument('--no_rotate', action='store_true',
                            help='Not to rotate image')
        parser.add_argument('--no_resize', action='store_true',
                            help='Not to resize image')
        parser.add_argument('--jittor', action='store_true',
                            help='Applying color adjustment during training')
        parser.add_argument('--crop_scale', type=float, default=0.75,
                            help='Scale of crop operation.')
        parser.add_argument('--save_freq', type=int, default=1,
                            help='Saving network states per epochs')
        parser.add_argument('--save_freq_step', type=int, default=5000,
                            help='Saving latest network states per steps')
        parser.add_argument('--print_state_freq', type=int, default=1000,
                            help='Print training states per iterations')

        # Discriminator settings
        parser.add_argument('--ndf', type=int, default=64,
                            help='Channel size base of Discriminator [64 | 96]')
        parser.add_argument('--n_layer', type=int, default=3,
                            help='Layer number of Discriminator')
        parser.add_argument('--use_spec', action='store_true',
                            help='Use spectral normalization in discriminator')
        return parser


    def mkdirs(self, opt):
        def mkdir(path):
            if not os.path.exists(path):
                os.mkdir(path)

        opt.ckpt_path = os.path.join(opt.save_path, opt.name)
        opt.sample_path = os.path.join(opt.ckpt_path, 'sample')
        opt.test_path = os.path.join(opt.ckpt_path, 'test')

        mkdir(opt.save_path)
        mkdir(opt.ckpt_path)
        mkdir(opt.sample_path)
        mkdir(opt.test_path)


    def parse(self, opt):
        if self.eval:
            opt.batch_size = 1

        str_ids = opt.gpus.split(',')
        opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpus.append(id)
        if len(opt.gpus) > 0:
            torch.cuda.set_device(opt.gpus[0])

        self.print_options(opt)


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        phase = 'train' if not self.eval else 'evaluation'
        mode = 'at' if not self.eval else 'wt'
        file_name = os.path.join(opt.ckpt_path, '{}_opt.txt'.format(phase))
        with open(file_name, mode) as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def get_options(self, eval=False):
        parser = self.initialize(eval)
        self.parser = self.modify_options(parser)
        opt = self.parser.parse_args()
        opt.eval = self.eval

        self.mkdirs(opt)
        self.parse(opt)

        return opt