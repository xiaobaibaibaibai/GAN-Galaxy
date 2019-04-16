from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

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

# Load galaxy image data
X_img = np.load("data/images.small.npy")
X_img = X_img.transpose([0,2,3,1])
X_img.shape

image_size = X_img.shape[1]
image_size

# Load targets
HSC_ids = np.load("data/HSC_ids.npy")
HSC_ids

df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]


df = df.drop_duplicates("HSC_id").set_index("HSC_id")[["photo_z", "log_mass"]]

df.head()

y = df.loc[HSC_ids].values

'''
y_for_visualization_samples = np.array([.14, 8.51])

standardizer = misc.Standardizer()
standardizer.train(y)
print("means: ", standardizer.means)
print("std:   ", standardizer.std)
y_standard = standardizer(y)
y_for_visualization_samples_standard = standardizer(y_for_visualization_samples)

y_standard.shape


# Run GAN
num_threads = 10

sess = tf.Session(config=tf.ConfigProto(
    intra_op_parallelism_threads=num_threads,
    inter_op_parallelism_threads=num_threads,
))

train = True
if train:
    # num_epochs = 450
    num_epochs = 10
    # use a dir outside of dropbox
    checkpoint_dir = os.path.join(os.path.expanduser("~"),
                                  "tmp - models",
                                  "models/gan/checkpoints")
else:
    num_epochs = 1
    # use a dir inside the repo
    checkpoint_dir = "models/gan/checkpoints"

batch_size = 64
z_dim = 100
dataset_name = "galaxy"
result_dir = "models/gan/results"
log_dir = "models/gan/log"

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



'''

