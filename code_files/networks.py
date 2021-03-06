import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch

def cgan_earth_nets(path, Training, g_dim, d_dim):
    # g_dim and d_dim are input dimensions into d and g

    hidden_dim = 64
    if Training == True:
        layers_g = [g_dim, hidden_dim*32, hidden_dim*16, hidden_dim*8, hidden_dim*8, hidden_dim*8, 3]
        kernel_g = [4, 4, 4, 4, 4, 3]
        stride_g = [2, 2, 2, 2, 2, 1]
        pad_g = [2, 2, 2, 2, 2, 0]
        layers_d = [d_dim, hidden_dim*8, hidden_dim*8, hidden_dim*8, hidden_dim*16, hidden_dim*32, 1]
        kernel_d = [4, 4, 4, 4, 4, 4]
        stride_d = [2, 2, 2, 2, 2, 1]
        pad_d = [1, 1, 1, 1, 1, 0]
        params = [layers_g, kernel_g, stride_g, pad_g, layers_d, kernel_d, stride_d, pad_d]
        with open(path + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)     

    class Generator(nn.Module):
        '''
        Generator Class
        Values:
            z_dim: the dimension of the noise vector, a scalar
            im_chan: the number of channels in the images
            hidden_dim: the inner dimension, a scalar
        '''
        def __init__(self, g_dim, img_length, im_chan=3, hidden_dim=64):
            super(Generator, self).__init__()
            self.img_length = img_length
            self.final_conv = nn.Conv2d(hidden_dim * 8, im_chan, 3, 1, 0)
            # Build the neural network
            self.gen = nn.Sequential(
                self.make_gen_block(g_dim, hidden_dim * 32),
                self.make_gen_block(hidden_dim * 32, hidden_dim * 16),
                self.make_gen_block(hidden_dim * 16, hidden_dim * 8),
                self.make_gen_block(hidden_dim * 8, hidden_dim * 8),
                self.make_gen_block(hidden_dim * 8, hidden_dim * 8)
            )

        def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=2):
            '''
            Function to return a sequence of operations corresponding to a generator block of DCGAN;
            a transposed convolution, a batchnorm (except in the final layer), and an activation.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
            '''
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, noise, labels, Training=True, ratio=2):
            '''
            Function for completing a forward pass of the generator: Given a noise tensor,
            returns generated images.
            Parameters:
                noise: a noise tensor with dimensions (n_samples, z_dim)
            '''
            x = torch.cat((noise.float(), labels.float()), 1)
            for layer in self.gen:
                x = layer(x)
            return torch.sigmoid(self.final_conv(x))


    class Critic(nn.Module):
        '''
        Critic Class
        Values:
            im_chan: the number of channels in the images, fitted for the dataset used
            hidden_dim: the inner dimension, a scalar
        '''
        def __init__(self, d_dim, hidden_dim=64):
            super(Critic, self).__init__()
            self.crit = nn.Sequential(
                self.make_crit_block(d_dim, hidden_dim * 8),
                self.make_crit_block(hidden_dim * 8, hidden_dim * 8),
                self.make_crit_block(hidden_dim * 8, hidden_dim * 8),
                self.make_crit_block(hidden_dim * 8, hidden_dim * 16),
                self.make_crit_block(hidden_dim * 16, hidden_dim * 32),
                self.make_crit_block(hidden_dim * 32, 1, stride=1, final_layer=True),
            )

        def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
            '''
            Function to return a sequence of operations corresponding to a critic block of DCGAN;
            a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
                final_layer: a boolean, true if it is the final layer and false otherwise 
                        (affects activation and batchnorm)
            '''
            if not final_layer:
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding=0),
                )

        def forward(self, image, labels):
            '''
            Function for completing a forward pass of the critic: Given an image tensor, 
            returns a 1-dimension tensor representing fake/real.
            Parameters:
                image: a flattened image tensor with dimension (im_chan)
            '''
            x = torch.cat((image.float(), labels.float()), 1)
            for layer in self.crit:
                x = layer(x)
            return x.view(len(x), -1)

    return Generator, Critic