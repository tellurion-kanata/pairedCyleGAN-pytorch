import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

import os
import numpy.random as random
from glob import glob
import PIL.Image as Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transform_seeds(seed_range, img_size=512, crop_size=None):
    seed = random.randint(-seed_range, seed_range)
    if crop_size is None:
        return seed
    if crop_size == img_size:
        return seed, None
    top, left = random.randint(0, img_size - crop_size, 2)
    crops = [top, left, crop_size]

    return seed, crops


def custom_transform(img, seed, opt, crops=None, is_ref=True):
    if not opt.no_rotate and is_ref:
        img = tf.rotate(img, seed, fill=255)
    if not opt.no_flip and seed >= 0:
        img = tf.hflip(img)
    if not opt.no_crop and crops is not None:
        top, left, length = crops[:]
        img = tf.crop(img, top, left, length, length)
    if not opt.no_resize:
        img = tf.resize(img, opt.load_size)
    return img


def jitter(img, seeds):
    brt, crt, sat = seeds[:]
    img = tf.adjust_brightness(img, brt)
    img = tf.adjust_contrast(img, crt)
    img = tf.adjust_saturation(img, sat)
    return img


def normalize(img, grayscale=False):
    img = transforms.ToTensor()(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img


class DraftDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.eval = opt.eval

        image_pattern = opt.image_pattern
        if not os.path.exists(self.opt.dataroot):
            raise FileNotFoundError('data file is not found.')
        self.sketch_root = os.path.join(self.opt.dataroot, 'sketch')
        self.color_root = os.path.join(self.opt.dataroot, 'color')
        self.reference_root = os.path.join(self.opt.dataroot, 'reference')
        self.image_files = glob(os.path.join(self.color_root, image_pattern))
        if not self.eval:
            self.image_size = opt.image_size                                                    # image size in training dataset
            self.crop_size = max(self.opt.load_size, round(self.image_size * opt.crop_scale))


    def __getitem__(self, index):
        image_file = self.image_files[index]
        seed = random.randint(0, len(self))
        skt_file = self.image_files[seed]

        color = Image.open(image_file).convert('RGB')
        sketch = Image.open(skt_file.replace(self.color_root, self.sketch_root)).convert('L')

        if not self.eval:
            # flip, crop and resize in custom transform function
            seed, crops = get_transform_seeds(1, self.image_size, self.crop_size)
            sketch = custom_transform(sketch, seed, self.opt, crops, is_ref=False)

            # no crop for reference image
            seed = get_transform_seeds(90)
            color = custom_transform(color, seed, self.opt)

            # change brightness, contrast and saturation in jitter function
            if self.opt.jittor:
                seed_j = random.random(3) * 0.2 + 0.9
                color = jitter(color, seed_j)

            sketch = normalize(sketch, grayscale=True)
            color = normalize(color)

            return {
                'sketch': sketch,
                'color': color,
                'index': index
            }

        else:
            ref = Image.open(image_file.replace(self.color_root, self.reference_root)).convert('RGB')
            if self.opt.resize:
                sketch = transforms.Resize((self.opt.load_size, self.opt.load_size))(sketch)
            h, w = sketch.size
            ref = transforms.Resize((h, w))(ref)
            index = image_file.replace('.jpg', '').replace('.png', '').replace(self.color_root, '').replace('/', '').replace('\\', '')

            sketch = normalize(sketch, grayscale=True)
            color = normalize(color)
            ref = normalize(ref)

            return {
                'sketch': sketch,
                'ref': ref,
                'color': color,
                'index': index
            }

    def __len__(self):
        return len(self.image_files)


class CustomDataLoader():
    def initialize(self, opt):
        self.dataset = DraftDataset(opt)

        self.dataLoader = data.DataLoader(
            dataset = self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.no_shuffle and not opt.eval,
            num_workers = opt.num_threads)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataLoader:
            yield data


class EvalDataset(data.Dataset):
    def __init__(self, dataroot, pattern='*.png'):
        self.dataroot = dataroot
        if not os.path.exists(self.dataroot):
            raise FileNotFoundError('data file is not found.')
        self.A_root = os.path.join(self.dataroot, 'fake')
        self.B_root = os.path.join(self.dataroot, 'reference')
        self.image_files = glob(os.path.join(self.A_root, pattern))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        A = Image.open(image_file).convert('RGB')
        B = Image.open(image_file.replace(self.A_root, self.B_root)).convert('RGB')

        A = normalize(A)
        B = normalize(B)
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.image_files)

class EvalDataLoader():
    def initialize(self, dataroot, batch_size, num_threads=16, pattern='*.png'):
        self.dataset = EvalDataset(dataroot, pattern)
        self.dataLoader = data.DataLoader(
            dataset = self.dataset,
            batch_size = batch_size,
            num_workers = num_threads)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataLoader:
            yield data