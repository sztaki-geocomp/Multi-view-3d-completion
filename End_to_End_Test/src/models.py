import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator,  Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, tv_loss
import cv2
from torch.nn.utils import clip_grad_norm_,  clip_grad_norm
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, self.gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, self.dis_weights_path)

class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        self.n_views = len(config.views)
        generator = InpaintGenerator(self.n_views)
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')

        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.SmoothL1Loss()
        cross_loss = nn.BCELoss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('cross_loss', cross_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks, GTs, n_views):
        self.iteration += 1

        outputs = self(images, masks, n_views)
        gen_loss = 0
        dis_loss = 0

        GTs = GTs.view(GTs.shape[0], n_views*3, 256, 256)
        masks = masks.view(masks.shape[0], n_views * 3, 256, 256)

        dis_input_real = GTs
        dis_input_fake = outputs.detach()

        for i in range(0, n_views):
            dis_real, _ = self.discriminator(dis_input_real[:, (3*i): (3*i) + 3, :, :])
            dis_fake, _ = self.discriminator(dis_input_fake[:, (3*i): (3*i) + 3, :, :])
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake = outputs
            gen_fake, _ = self.discriminator(gen_input_fake[:, (3*i): (3*i) + 3, :, :])
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss/n_views

            gen_content_loss = self.perceptual_loss(outputs[:, (3 * i): (3 * i) + 3, :, :],
                                                    GTs[:, (3 * i): (3 * i) + 3, :, :])
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss / n_views

            gen_style_loss = self.style_loss(
                outputs[:, (3 * i): (3 * i) + 3, :, :] * masks[:, (3 * i): (3 * i) + 3, :, :],
                GTs[:, (3 * i): (3 * i) + 3, :, :] * masks[:, (3 * i): (3 * i) + 3, :, :])
            gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            gen_loss += gen_style_loss / n_views

            #grad_real = imgrad(GTs[:, (3 * i): (3 * i) + 1, :, :])
            #grad_fake = imgrad(outputs[:, (3*i): (3*i) + 1, :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #gen_loss += 10 * grad_loss/21

            #grad_real = imgrad(GTs[:, (3 * i)+1: (3 * i) + 2, :, :])
            #grad_fake = imgrad(outputs[:, (3*i)+1: (3*i) + 2, :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #gen_loss += 10 * grad_loss/21

            #grad_real = imgrad(GTs[:, (3 * i)+2: (3 * i) + 3, :, :])
            #grad_fake = imgrad(outputs[:, (3*i)+2: (3*i) + 3, :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #gen_loss += 10 * grad_loss/21
            #TV1 = tv_loss(outputs[:, (3*i): (3*i) + 3, :, :])
            #gen_loss += (TV1 / 65536)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)


            #gen_loss += 0.1 * normal_loss/21

            #grad_real = imgrad(GTs[:, (5*i) + i +4 : ((5*i) + i) +5 , :, :])
            #grad_fake = imgrad(outputs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #gen_loss +=  10* grad_loss/21

            #print("10 * grad_loss/21 :", 10 * grad_loss.item()/21)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)
            #TV1 = tv_loss(outputs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #gen_loss += (TV1 / 65536)

            #print("(TV1 / 65536) * 2: " ,(TV1.item() / 65536) * 2)


            #gen_loss += 0.1 * normal_loss/21

            #grad_real = imgrad(GTs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #grad_fake = imgrad(outputs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)
            #gen_loss +=  10* grad_loss/21


            #TV1 = tv_loss(outputs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #gen_loss += (TV1 / 65536)
            #gen_loss += 0.1 * normal_loss/21

            #gen_content_loss = self.perceptual_loss(outputs[:, (5 * i) + i + 3: ((5 * i) + i) + 6, :, :],
            #                                        GTs[:, (5 * i) + i + 3: ((5 * i) + i) + 6, :, :])
            #gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            #print(gen_content_loss, gen_content_loss.item())

            #gen_loss += gen_content_loss/7


            # print (depth_loss)


            #gen_style_loss = self.style_loss(outputs[:, (5*i) + i +3 : ((5*i) + i) +6 , :, :] * masks[:, (5*i) + i +3: ((5*i) + i) +6 , :, :] , GTs[:, (5*i) + i+3 : ((5*i) + i) +6 , :, :] * masks[:, (5*i) + i+3 : ((5*i) + i) +6 , :, :])
            #gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            #gen_loss += gen_style_loss/7


        #print("grad_loss", grad_loss.item())
        #print("normaL_loss", normal_loss.item())
        #print("gen_style_loss", gen_style_loss.item())
        #print("gen_content_loss", gen_content_loss.item())





        # generator l1 loss
        #print ("output. shape :" , outputs.shape)
        #print("GTs. shape :", GTs.shape)


        gen_l1_loss = self.l1_loss(outputs, GTs) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        #print ("L1: ", gen_l1_loss.item()* self.config.L1_LOSS_WEIGHT / torch.mean(masks)/7)
        gen_loss += gen_l1_loss/n_views
        #print("rwara", gen_l1_loss_test)
        #gen_loss += gen_l1_loss_test

        Gts_binary = (GTs>0.000001).float()
        outputs_binary = (outputs> 0.000001).float()

        gen_cross_loss = self.cross_loss(outputs_binary, Gts_binary) * 0.1
        #print("gen_cross_loss: ", gen_cross_loss.item()/7)
        gen_loss += gen_cross_loss/n_views

        #gen_loss += dist_forward









        # generator perceptual loss
        #print (outputs[:, 0:3 , :, :].shape)

        ##masks = masks.view(masks.shape[0], 7 * 6, 256, 256)

        ##for i in range (0,7):
            #print (i)

            #depth_loss = self.depth_criterion(outputs[:, (5*i) + i +3 : ((5*i) + i) +4 , :, :], GTs[:, (5*i) + i +3 : ((5*i) + i) +4 , :, :])

            #grad_real = imgrad_yx(GTs[:, (5*i) + i +3 : ((5*i) + i) +4 , :, :])
            #grad_fake = imgrad_yx(outputs[:, (5*i) + i +3 : ((5*i) + i) +4 , :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)

            #print (depth_loss)
            #print(grad_loss)
            #print(normal_loss)

            #gen_loss += depth_loss/7
            #gen_loss += 0.1* grad_loss/7
            #gen_loss += 0.1* normal_loss/7

            #grad_real = imgrad_yx(GTs[:, (5*i) + i +4 : ((5*i) + i) +5 , :, :])
            #grad_fake = imgrad_yx(outputs[:, (5*i) + i +4 : ((5*i) + i) +5 , :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)

            #print (depth_loss)
            #print("grad_loss",grad_loss)
            #print("normaL_loss",normal_loss)
            #depth_loss = self.depth_criterion(outputs[:, (5 * i) + i + 4: ((5 * i) + i) + 5, :, :],
            #                                  GTs[:, (5 * i) + i + 4: ((5 * i) + i) + 5, :, :])

            #gen_loss += depth_loss/7
            #gen_loss += 0.1* grad_loss/7
            #gen_loss += 0.1* normal_loss/7

            #grad_real = imgrad_yx(GTs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #grad_fake = imgrad_yx(outputs[:, (5*i) + i +5 : ((5*i) + i) +6 , :, :])
            #grad_loss = self.grad_criterion(grad_fake, grad_real)
            #normal_loss = self.normal_criterion(grad_fake, grad_real)

            #print (depth_loss)
            #print(grad_loss)
            #print(normal_loss)
            #depth_loss = self.depth_criterion(outputs[:, (5 * i) + i + 5: ((5 * i) + i) + 6, :, :],
            #                                  GTs[:, (5 * i) + i + 5: ((5 * i) + i) + 6, :, :])

            #gen_loss += depth_loss/7
            #gen_loss += 0.1*grad_loss/7
            #gen_loss += 0.1* normal_loss/7

            ##gen_content_loss = self.perceptual_loss(outputs[:, (5*i) + i : ((5*i) + i) +3 , :, :], GTs[:, (5*i) + i : ((5*i) + i) +3 , :, :])
            ##gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            ##gen_loss += gen_content_loss/7


            ##gen_style_loss = self.style_loss(outputs[:, (5*i) + i : ((5*i) + i) +3 , :, :] * masks[:, (5*i) + i : ((5*i) + i) +3 , :, :] , GTs[:, (5*i) + i : ((5*i) + i) +3 , :, :] * masks[:, (5*i) + i : ((5*i) + i) +3 , :, :])
            ##gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            ##gen_loss += gen_style_loss/7

            #gen_content_loss = self.perceptual_loss(outputs[:, (5*i) + i +3 : ((5*i) + i) +6 , :, :], GTs[:, (5*i) + i +3 : ((5*i) + i) +6 , :, :])
            #gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            #gen_loss += gen_content_loss/7


            #gen_style_loss = self.style_loss(outputs[:, (5*i) + i +3 : ((5*i) + i) +6 , :, :] * masks[:, (5*i) + i +3: ((5*i) + i) +6 , :, :] , GTs[:, (5*i) + i+3 : ((5*i) + i) +6 , :, :] * masks[:, (5*i) + i+3 : ((5*i) + i) +6 , :, :])
            #gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
            #gen_loss += gen_style_loss/7


        # generator style loss





        # create logs
        logs = [
            #("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            #("l_per", gen_content_loss.item()),
            #("l_sty", gen_style_loss.item()),
        ]

        #return outputs, gen_loss, dis_loss, logs
        #dis_loss = None
        return outputs, gen_loss, dis_loss, logs

    def forward(self, images,  masks,N_views):

        #print (images.shape)
        #print (masks.shape)

        #print(images.shape)
        #print(masks.shape)
        #print(edges.shape)

        images_masked = (images * (1 - masks).float()) + masks
        #depths_masked = (depths * (1 - masks).float()) + masks
        #inputs = torch.cat((images_masked,  label), dim=2)
        #print ("inputs.shape: ", inputs.shape)
        #outputs = self.generator(inputs, label)                                    # in: [rgb(3) + edge(1)]
        outputs = self.generator(images_masked,N_views)
        return outputs



    def backward(self, gen_loss=None, dis_loss=None, iterations=None):
        batch_size = 4


        #self.gen_optimizer.zero_grad()
        #self.dis_optimizer.zero_grad()

        if gen_loss is not None:
            (gen_loss/batch_size).backward(retain_graph=True)

        if dis_loss is not None:
            (dis_loss/batch_size).backward(retain_graph=True)

        if (iterations % batch_size ==0):
            self.gen_optimizer.step()
            self.dis_optimizer.step()

            self.gen_optimizer.zero_grad()
            self.dis_optimizer.zero_grad()

        '''
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
        
        '''

