import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns

import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.figsize'] = np.array((10,6))*.6

# my code
import misc
import gan



img_dir = "data/galaxy_images_training/npy_files/"
filename_formatter = os.path.join(img_dir, "{}-cutout.npy")

npy_files = glob.glob(filename_formatter.format("*"))

HSC_ids = np.array([int(os.path.basename(f).split("-")[0]) for f in npy_files])

print("HSC_ids")


X_img = np.empty([len(HSC_ids), 3, 50, 50])
for i, HSC_id in enumerate(HSC_ids):
    X_img[i] = np.load(filename_formatter.format(HSC_id))

X_img = X_img.transpose([0,2,3,1])
print("X_img.shape")

image_size = X_img.shape[1]
print("image_size")


df = pd.read_csv("data/2018_02_23-all_objects.csv")
# df = df[df.selected]


df = df.drop_duplicates("HSC_id").set_index("HSC_id")[["photo_z", "log_mass"]]

print("df.head()")


y = df.loc[HSC_ids].values
y_for_visualization_samples = np.array([.14, 8.51])


# values copied from output of `simple gan.ipynb`
standardizer = misc.Standardizer(means = np.array([0.21093612, 8.62739865]),
                                 std = np.array([0.30696933, 0.63783586]))
# standardizer.train(y)
print("means: ", standardizer.means)
print("std:   ", standardizer.std)
y_standard = standardizer(y)
y_for_visualization_samples_standard = standardizer(y_for_visualization_samples)

print("y_standard.shape")



num_threads = 4

sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=num_threads,
    inter_op_parallelism_threads=num_threads,
))

train = True
if train:
    num_epochs = 100
    # use a dir outside of dropbox
    checkpoint_dir = os.path.join(os.path.expanduser("~"),
                                  "GAN-Galaxy",
                                  "models/all_gan/checkpoints")
else:
    num_epochs = 1
    # use a dir inside the repo
    checkpoint_dir = "models/gan/checkpoints"
    
batch_size = 64
z_dim = 100
dataset_name = "galaxy_all"
result_dir = "models/all_gan"
log_dir = "models/all_gan/log"

model = gan.CGAN(sess, num_epochs, batch_size, z_dim, dataset_name,
                 image_size, X_img, 
                 y_standard, y_for_visualization_samples_standard,
                 checkpoint_dir, result_dir, log_dir,
                 d_learning_rate=.0001,
                 relative_learning_rate=4.,
                 loss_weighting=50.,
                )

model.build_model()

model.train()
