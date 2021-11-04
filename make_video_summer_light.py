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

'''
with open(proj_path + '_noise.data', 'rb') as filehandle:
    # store the data as binary data stream
    noise = pickle.load(filehandle)


imgs1, noise, netG = roll_video(proj_path, desert_lbl, netG(z_dim+n_classes, img_length), n_classes, z_dim, lf, device, ratio, n_clips=int(8/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs1.shape)
imgs2, noise, netG = transit_video(desert_lbl, star_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_size=0.5, l_step_size=0.5, transit_mode='scroll')
print(imgs2.shape)
imgs4, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=3)
print(imgs4.shape)
imgs5, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(4/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs5.shape)
imgs6, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=5)
print(imgs6.shape)
imgs7, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(8/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs7.shape)
imgs8, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=3)
print(imgs8.shape)
imgs9, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(4/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs9.shape)
imgs10, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=5)
print(imgs10.shape)
imgs11, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(8/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs11.shape)
imgs12, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=3)
print(imgs12.shape)
imgs13, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(4/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs13.shape)
imgs14, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=5)
print(imgs14.shape)
imgs15, noise, netG = roll_video(proj_path, star_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(4/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs15.shape)
imgs16, noise, netG = effects(star_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_num=2, l_step_size=1, z_max_num=4, effect='circles', n_circles=5)
print(imgs16.shape)
'''

# section s1
imgs1, noise, netG = roll_video(proj_path, desert_lbl, netG(z_dim+n_classes, img_length), n_classes, z_dim, lf, device, ratio, n_clips=int(224/(1/0.25)), step_size=0.25)
print(imgs1.shape)
imgs2, noise, netG = transit_video(desert_lbl, sea_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.25, z_step_size=0.25, l_step_size=0.2, transit_mode='scroll')
print(imgs2.shape)
imgs3, noise, netG = roll_video(proj_path, sea_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(160/(1/0.25)), step_size=0.25, original_noise=noise)
print(imgs3.shape)
imgs4, noise, netG = transit_video(sea_lbl, forest_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.5, z_step_size=0.2, l_step_size=0.2, transit_mode='circular')
print(imgs4.shape)
imgs5, noise, netG = roll_video(proj_path, forest_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(182/(1/0.5)), step_size=0.5, original_noise=noise)
print(imgs5.shape)
imgs6, noise, netG = transit_video(forest_lbl, desert_lbl, n_classes, noise, netG, lf, ratio, device, step_size=0.5, z_step_size=0.1, l_step_size=0.1, transit_mode='uniform')
print(imgs6.shape)
imgs7, noise, netG = roll_video(proj_path, desert_lbl, netG, n_classes, z_dim, lf, device, ratio, n_clips=int(184/(1/0.5)), step_size=0.5, original_noise=noise)
print(imgs7.shape)


# concatenante the imgs together and make video
imgs = np.vstack((imgs1, imgs2))
#imgs = np.vstack((imgs, imgs3))
imgs = np.vstack((imgs, imgs4))
imgs = np.vstack((imgs, imgs5))
imgs = np.vstack((imgs, imgs6))
imgs = np.vstack((imgs, imgs7))
#imgs = np.vstack((imgs, imgs8))
#imgs = np.vstack((imgs, imgs9))
#imgs = np.vstack((imgs, imgs10))
#imgs = np.vstack((imgs, imgs11))
#imgs = np.vstack((imgs, imgs12))
#imgs = np.vstack((imgs, imgs13))
#imgs = np.vstack((imgs, imgs14))
#imgs = np.vstack((imgs, imgs15))
#imgs = np.vstack((imgs, imgs16))
#imgs = np.vstack((imgs, imgs17))
#imgs = np.vstack((imgs, imgs18))
#imgs = np.vstack((imgs, imgs19))
animate(proj_path, imgs, fps=25)

with open(proj_path + '_noise.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(noise, filehandle) 
