import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from skimage.color import rgb2gray, gray2rgb
import cv2
from PIL import Image

class MultiView():
    def __init__(self, config):
        self.config = config
        model_name = 'inpaint'
        self.debug = False
        self.model_name = model_name
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.N_views = len(config.views)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False,
                                        training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True,
                                         training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.inpaint_model.load()

    def save(self):
        self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        gradient_accumulations = 4

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.inpaint_model.train()

                images, masks, GTs = self.cuda(*items)
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, GTs, self.N_views)
                masks = masks.view(masks.shape[0], self.N_views * 3, 256, 256)
                images = images.view(images.shape[0], self.N_views * 3, 256, 256)
                GTs = GTs.view(GTs.shape[0], self.N_views * 3, 256, 256)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                psnr = self.psnr(self.postprocess(GTs), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(GTs - outputs_merged)) / torch.sum(GTs)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))
                del psnr
                del mae

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss, self.inpaint_model.iteration)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(len(images),
                            values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.val_dataset)

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, masks, GTs = self.cuda(*items)

            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, GTs, self.N_views)
            masks = masks.view(masks.shape[0], self.N_views * 3, 256, 256)
            images = images.view(images.shape[0], self.N_views * 3, 256, 256)
            GTs = GTs.view(GTs.shape[0], self.N_views * 3, 256, 256)
            outputs_merged = (outputs * masks) + (images * (1 - masks))
            psnr = self.psnr(self.postprocess(GTs), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(GTs - outputs_merged)) / torch.sum(GTs)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))
            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):

        self.inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name_folader = self.test_dataset.load_name(index)
            print(name_folader, "Done")
            path = os.path.join(self.results_path, name_folader)
            if not os.path.exists(path):
                os.makedirs(path)
            images, masks, GTs = self.cuda(*items)
            index += 1

            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, images, self.N_views)

            images = images.view(images.shape[0], self.N_views * 6, 256, 256)
            masks = masks.view(masks.shape[0], self.N_views * 6, 256, 256)

            outputs_merged = (outputs * masks) + (images * (1 - masks))
            # outputs_merged = outputs

            for i in range(self.N_views):

                output = self.postprocess(outputs_merged)[:, :, 6 * i: 6 * i + 3]
                name = 'color_{0:03d}.png'.format(i)
                path = os.path.join(self.results_path, name_folader, name)
                im = Image.fromarray(output)
                im.save(path)
                #imsave(output, path)
                #cv2.imwrite(path, output)


                output = self.postprocess_65536(outputs_merged)[:, :, 6 * i+3: 6 * i + 6]
                name = 'locations_{0:03d}.png'.format(i)
                path = os.path.join(self.results_path, name_folader, name)
                cv2.imwrite(path, output)

                #output = self.postprocess_65536(outputs)[:, :, 3 * i: 3 * i + 3]
                #name = 'gan_locations_{0:03d}.png'.format(i)
                #path = os.path.join(self.results_path, name_folader, name)
                #cv2.imwrite(path, output)

                torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        print('\nEnd Completion model test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.inpaint_model.eval()

        model = self.config.MODEL
        items = next(self.sample_iterator)
        images, masks, GTs = self.cuda(*items)

        iteration = self.inpaint_model.iteration

        inputs = (images * (1 - masks)) + masks
        outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, masks, GTs, self.N_views)

        images = images.view(images.shape[0], self.N_views * 3, 256, 256)
        masks = masks.view(masks.shape[0], self.N_views * 3, 256, 256)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        inputs = inputs.view(inputs.shape[0], self.N_views * 3, 256, 256)
        GTs = GTs.view(GTs.shape[0], self.N_views * 3, 256, 256)
        outputs = outputs_merged

        if it is not None:
            iteration = it

        images = stitch_images(

            self.postprocess(inputs),
            self.postprocess(outputs),
            self.postprocess(GTs),
            n_views=self.N_views
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):

        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        img = img.cpu().detach().squeeze().numpy().astype(np.uint8)
        return img

    def postprocess_65536(self, img):

        img = img * 65536.0
        img = img.permute(0, 2, 3, 1)
        img = img.cpu().detach().squeeze().numpy().astype(np.uint16)

        return img

