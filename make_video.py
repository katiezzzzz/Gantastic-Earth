from code_files import *
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
Project_name = 'earth_cylinder_t_12'
Project_dir = PATH + '/trained_generators/'
wandb_name = Project_name

labels = [0, 1, 2, 3]

# define hyperparameters and architecture
ngpu = 1
z_dim = 64
lr = 0.0001
Training = 0
n_classes = 4
batch_size = 8
im_channels = 3
num_epochs = 600
img_length = 128 # size of training image
proj_path = mkdr(Project_name, Project_dir, Training)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

# create networks
netG, netD = cgan_earth_nets(proj_path, Training, z_dim+n_classes, im_channels+n_classes)

# animate
forest_lbl = [0]
desert_lbl = [1]
sea_lbl = [2]
star_lbl = [3]
lf = 28
ratio = 5

# section s1
imgs1, noise, netG = roll_video(proj_path, desert_lbl, netG(z_dim+n_classes, img_length), n_classes, z_dim, lf, device, ratio, n_clips=int(225/(1/0.25)), step_size=0.25)
print(imgs1.shape)
imgs2, noise, netG = transit_video(desert_lbl, sea_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_size=0.25, l_step_size=0.2, transit_mode='scroll')
print(imgs2.shape)

# concatenante the imgs together and make video
imgs = np.vstack((imgs1, imgs2))
animate(proj_path, imgs, fps=25)
