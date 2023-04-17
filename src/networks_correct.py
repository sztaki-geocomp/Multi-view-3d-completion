import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, N_views, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        #self.label_embedding = nn.Embedding(7, ([10, 64, 64]))

        self.N_views = N_views

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3*self.N_views, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))
        self.encoder2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True))


        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        #self.rnn = nn.RNN(256 * 64 * 64, 256, 7, batch_first = True)

        #self.middle_all = nn.Sequential(


        #    nn.Conv2d(in_channels=self.N_views*256, out_channels=256, kernel_size=3, stride=1, padding=1),
        #    nn.InstanceNorm2d(256, track_running_stats=False),
        #    nn.ReLU(True)
        #)


        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            #nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))
        self.decoder2 = nn.Sequential(nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, padding=0),
            nn.Conv2d(in_channels=64, out_channels=3 * self.N_views, kernel_size=3, padding=1),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x, N_views):
        #print(x.shape)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.middle(x2)
        #x2 = torch.inverse(x2)
        #x2 = torch.flip(x2,[0])
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxy1 = torch.cat([x3, x2[::-1,:,:]], 1)
        #y1 = torch.cat([x3, x2], 1)

        #print(y1.shape)
        y1 = self.decoder1(x3)
        #x1 = torch.inverse(x1)
        ###x1 = torch.flip(x1,[0])
        #y2 = torch.cat([y1, x1[::-1,:,:]], 1)
        ###y2  = torch.cat([y1, x1], 1)
        y2 = self.decoder2(y1)
        y2 = (torch.tanh(y2) + 1) / 2
        #print(y2.shape)
        return y2





    #def forward(self, x):

        #tmp = torch.zeros(x.shape[0], x.shape[1], 256, 64, 64).cuda()

        '''
        for i in range(7):
            img = x[ :, i, :, :, :]
            #print (img.shape)
            img = self.encoder(img)
            img = self.middle(img)
            #print (img.shape)
            #tmp[:, i] = img
            if i==0 :
                tmp = torch.unsqueeze(img,1)
            else:
                #img = torch.unsqueeze(img,4)
                tmp = torch.cat([tmp, torch.unsqueeze(img,1)], 1)
            del img



        

        for i in range(N_views):
            #img = x[:, i, :, :, :]
            # print (img.shape)
            #x[:, i, :, :, :] = torch.unsqueeze(self.middle(self.encoder(x[:, i, :, :, :])), 1)
            #x[:, i, :, :, :] =
            # print (img.shape)
            # tmp[:, i] = img
            if i == 0:
                tmp = torch.unsqueeze(self.middle(self.encoder(x[:, i, :, :, :])), 1)
            else:
                # img = torch.unsqueeze(img,4)
                tmp = torch.cat([tmp, torch.unsqueeze(self.middle(self.encoder(x[:, i, :, :, :])), 1)], 1)
            #del img

        #print (tmp.shape)

        #mean_tmp = torch.mean(tmp, 1, False)

        #print(tmp.shape)
        tmp1 = tmp.view(tmp.shape[0], N_views * 256, tmp.shape[-2], tmp.shape[-1])

        #print(tmp1.shape)
        #mean_tmp = torch.mean(tmp, 1, False)
        mean_tmp = self.middle_all(tmp1)

        #print(mean_tmp.shape)
        '''


        """
                for i in range(N_views):
            #img = tmp[:, i, :, :, :]
            #label = labels[:, i, :, :, :]
            #print(label.shape)

            #x = torch.cat([ tmp[:, i, :, :, :], mean_tmp, labels[:, i, :, :, :]], 1)
            x = torch.cat([tmp[:, i, :, :, :], mean_tmp], 1)
            
            #del img
            #del label
            #print (in_decoder.shape)
            x = self.decoder(x)
            x = (torch.tanh(x) + 1) / 2
            #print('out.shape ', x.shape)
            if i==0 :
                tmp_out = x
            else:
                tmp_out = torch.cat([tmp_out, x], 1)
            #tmp_out[:, i*4: i*4+4, :, :] = x
        """


        #del tmp
        #del mean_tmp
        #del x





            


            #tmp[:, i] = img
            #del img






        #print ("Done: ")



            #pass





        #x = self.middle(x)
        #c = self.label_embedding(labels)
        #print(x.shape)
        #print(labels.shape)

        #x = torch.cat([x, labels], 1)
        #print (x.shape)

        #x = self.decoder(x)
        #x = (torch.tanh(x) + 1) / 2

        #x = x.reshape(x.shape[0], 28, 256, 256)

        #return tmp_out

"""
class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
"""



class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )



        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):

        #print (labels.shape)
        
        #print(x.shape)
        #labels = labels.view(x.shape[0], -1, 64, 64)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        #labels = labels [:,:,:31,:31]
        #print(labels.shape)
        #conv4 = torch.cat([conv4, labels], 1)
        #print (conv4.shape)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
