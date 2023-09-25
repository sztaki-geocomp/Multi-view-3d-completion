import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from skimage.color import rgb2gray, gray2rgb
import cv2
from imageio import imread

#import scipy
#import random
#from skimage.feature import canny

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist,  mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)
        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK
        self.nms = config.NMS
        self.sample_matrix = config.samples
        self.sample_views = config.views
        self.N_samples = len(self.sample_matrix)
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 2

    def __len__(self):
        if self.training:
            return len(self.data) * self.N_samples
        else:
            return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        if self.training:
            return os.path.basename(name)
        else:
            return '/'.join(name.split('/')[-2:])

    def load_item(self, index):
        n_views = len(self.sample_views)
        sample_ind = index % self.N_samples
        sample = self.sample_matrix[sample_ind]

        all_in_out = torch.empty(size=(n_views, 6, 256, 256))
        mask_out = torch.empty(size=(n_views, 6, 256, 256))
        gt_out = torch.empty(size=(n_views, 6, 256, 256))


        if self.training == 1:
            for ind, i in enumerate(self.sample_views):

                # Load the geometry channels  for the Ground Truth
                depth_name = (self.data[int(index/self.N_samples)] + '/locations_{0:03d}.png'.format(i))
                depth_gt = cv2.imread(depth_name, -1)
                depth_gt = depth_gt.astype(float) / 65536
                # Load the color channels  for the Ground Truth
                img_gt = np.array(imread(depth_name.replace('locations', 'color')))
                img_gt = img_gt.astype(float) / 256
                gt = np.concatenate([img_gt, depth_gt], axis=2)
                gt = depth_gt

                # Load the geometry channels  for the Input
                depth_in = cv2.imread(self.data[int(index / self.N_samples)].replace('GT', 'in') + '/{0:1d}'.format(
                    sample) + '/locations_{0:03d}.png'.format(i), -1)
                depth_in = depth_in.astype(float) / 65536

                # Load the color channels for the Input
                img_in = np.array(imread(self.data[int(index / self.N_samples)].replace('GT', 'in') + '/{0:1d}'.format(
                    sample) + '/color_{0:03d}.png'.format(i)))
                img_in = img_in.astype(float) / 256
                all_in = np.concatenate([img_in, depth_in], axis=2)
                all_in = depth_in

                mask = self.load_mask(depth_in)
                mask1 = gray2rgb(mask)
                mask = np.concatenate([mask1, mask1], axis=2)
                #mask = mask1

                all_in_out[ind] = self.to_tensor(all_in)
                mask_out[ind] = self.to_tensor(mask)
                gt_out[ind] = self.to_tensor(gt)

                #all_in_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(all_in)
                #mask_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(mask)
                #gt_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(gt)

            return all_in_out,  mask_out, gt_out
        else:
            for ind, i in enumerate(self.sample_views):

                

                depth_in = cv2.imread(self.data[int(index)] + '/locations_{0:03d}.png'.format(i), -1)
                depth_in = depth_in.astype(float) / 65536
                img_in = np.array(imread(self.data[int(index)] + '/color_{0:03d}.png'.format(i)))
                img_in = img_in.astype(float) / 256
                all_in = np.concatenate([img_in, depth_in], axis=2)
                #all_in = depth_in

                mask = self.load_mask(depth_in)
                mask1 = gray2rgb(mask)
                #mask = mask1
                mask = np.concatenate([mask1, mask1], axis=2)


                all_in_out[ind] = self.to_tensor(all_in)
                mask_out[ind] = self.to_tensor(mask)
                


                #all_in_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(all_in)
                #mask_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(mask)
                #gt_out[ind * 3: (((ind + 1)*3))] = self.to_tensor(gt)

            return all_in_out,  mask_out

    def load_mask(self, img):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        if mask_type == 1:
            mask = rgb2gray(img)
            mask = (mask == 0).astype(np.uint8) * 255

            # If external masks are needed
            """
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask1 = imread(self.mask_data[mask_index])
            mask1 = self.resize(mask1, imgh, imgw)
            mask1 = (mask1 > 0).astype(np.uint8) * 255
            #mask_all = mask1 + mask
            """
            mask_all = mask

            mask_all = (mask_all > 0).astype(np.uint8) * 255

            return mask_all

        # test mode:
        if mask_type == 2:

            #mask = rgb2gray(img)
            #mask = (mask == 0).astype(np.uint8) * 255
            #mask = (mask > 0).astype(np.uint8) * 255
            #return mask

            mask1 = rgb2gray(img)
            mask1 = (mask1 == 0).astype(np.uint8) * 255
            mask = np.zeros((256, 256), dtype=int)
            mask_all = mask1 + mask
            mask_all = (mask_all > 0).astype(np.uint8) * 255
            return mask_all

    def to_tensor(self, img):
        #img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()

        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = np.array(Image.fromarray(img).resize((height, width)))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                # if the testing data is folder in folder
                flist = list(glob.glob(flist+'/*'+'/*'))
                # if the testing data is in folder
                # flist = list(glob.glob(flist+'/*'))

                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
