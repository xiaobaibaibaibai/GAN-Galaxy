import sys

arugments = sys.argv[1:]
if len(arugments) is not 2:
    print("only accept 2 arguements")
    sys.exit()
else:
    print("the redshift is {0}, the mass is {1}".format(arugments[0], arugments[1]))


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import matplotlib as mpl

import numpy as np
import pandas as pd
import os

import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras import backend as K
K.set_image_data_format('channels_first')

import seaborn as sns
sns.set(font_scale=2, style="ticks")

# my code
import misc
import gan

HSC_ids = np.load("data/HSC_ids.npy")


X = np.load("data/images.small.npy")

X_img = X.copy().transpose([0,2,3,1])


image_size = X.shape[-1]
image_shape = X.shape[1:]


df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]

df = df.drop_duplicates("HSC_id") \
       .set_index("HSC_id") \
       .loc[HSC_ids] \
       [["photo_z", "log_mass"]]
    

targets = (df.log_mass > 8) & (df.log_mass < 10) & (df.photo_z < .15)
print(targets.mean())
print(targets.sum())

y_conditionals = df.values


# y_conditionals_for_visualization = np.array([[.14, 8.51]])
y_conditionals_for_visualization = np.array([arugments[0], arugments[1]],dtype=np.float64)


meanNum = np.array([arugments[0], arugments[1]],dtype=np.float64)

# stdNum = np.array([arugments[0], arugments[1]],dtype=np.float64)
# stdNum[0] + 0.9
# stdNum[1] - 0.2

# values copied from output of `simple gan.ipynb`
# standardizer = misc.Standardizer(means = np.array([0.21093612, 8.62739865]),
#                                  std = np.array([0.30696933, 0.63783586]))
standardizer = misc.Standardizer(means = meanNum,
                                 std = np.array([0.30696933, 0.63783586]))
# standardizer.train(y)
print("means: ", standardizer.means)
print("std:   ", standardizer.std)

y_conditionals = standardizer(y_conditionals)
y_conditionals_for_visualization = standardizer(y_conditionals_for_visualization)


batch_size = 64


np.random.seed(seed=0)

randomized_indices = np.arange(X.shape[0])
np.random.shuffle(randomized_indices)

training_fraction = 0.8
# make sure training set size is an even multiple of 64
num_training = (int(training_fraction*X.shape[0]) // batch_size) * batch_size

training_set_indices = randomized_indices[:int(num_training)]
testing_set_indices = np.array(list(set([*randomized_indices]) - set([*training_set_indices])))


from keras.preprocessing.image import Iterator
from keras.preprocessing.image import array_to_img


class DAGANIterator(Iterator):
    """Iterator yielding data from a DAGAN
    # Arguments
        gan_model: conditional GAN object.
        y_target: Numpy array of targets data.
        y_conditional: Numpy array of conditionals data (for GAN)
            to do: it would be nice to allow this to be a generator.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        image_shape: array-like, length 3
            example: [3, 50, 50]
            required since I'm not passing any example images to this object
    """

    def __init__(self, gan_model, 
                 y_target, y_conditional,
                 batch_size=64, shuffle=False, seed=None,
                 data_format="channels_first",
                 save_to_dir=None, save_prefix='', save_format='png',
                 image_shape=None):
        if data_format is None:
            raise ValueError("`data_format` cannot be None.")
        self.gan_model = gan_model
        channels_axis = 3 if data_format == 'channels_last' else 1
        if y_target is not None:
            self.y_target = np.asarray(y_target)
        else:
            self.y_target = None
        if y_conditional is not None:
            self.y_conditional = np.asarray(y_conditional)
        else:
            self.y_conditional = None
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        if image_shape is None:
            raise ValueError("`image_shape` must be array-like of length 3")
        self.image_shape = image_shape
        
        if batch_size != self.gan_model.batch_size:
            raise ValueError("DAGANIterator batch_size must match self.gan_model.batch_size.")
        super(DAGANIterator, self).__init__(y_target.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):        
        y_conditionals = self.y_conditional[index_array]

        batch_x = self.gan_model.generate_samples(y_conditionals)
        
        batch_x = np.asarray(batch_x, dtype=K.floatx())
        batch_x = batch_x.transpose([0,3,1,2])
        
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y_target is None:
            return batch_x
        batch_y = self.y_target[index_array]
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


sess = tf.Session()

train = False
if train:
    num_epochs = 450
    # use a dir outside of dropbox
    checkpoint_dir = os.path.join(os.path.expanduser("~"),
                                  "tmp - models",
                                  "models/gan/checkpoints")
else:
    num_epochs = 1
    # use a dir inside the repo
    # checkpoint_dir = "models/gan/checkpoints"
    checkpoint_dir = "models/all_gan/checkpoints"

# batch_size = 64 # set above
z_dim = 100
# dataset_name = "galaxy"
dataset_name = "galaxy_all"
# result_dir = "models/gan/results"
result_dir = "models/classify_DAGAN/"
log_dir = "models/classify_DAGAN/log"

gan_model = gan.CGAN(sess, num_epochs, batch_size, z_dim, dataset_name,
                     image_size, X_img, 
                     y_conditionals, y_conditionals_for_visualization,
                     checkpoint_dir, result_dir, log_dir,
                     d_learning_rate=.0001,
                     relative_learning_rate=4.,
                    )

gan_model.build_model()
gan_model.train()



y_conditional_training = y_conditionals[training_set_indices]
y_target_training = targets.values[training_set_indices]

y_target_training.size


dagan_iterator = DAGANIterator(gan_model, y_target_training, y_conditional_training,
                               image_shape=image_shape, 
                               shuffle=True,
             )

batch_idx = np.arange(64)

y_conditionals_tmp = y_conditionals[batch_idx]

samples = gan_model.generate_samples(y_conditionals_tmp)

plt.imshow(misc.transform_0_1(samples[0]))
plt.savefig('result.png')

n_conv_filters = 16
conv_kernel_size = 4
input_shape = X.shape[1:]

dropout_fraction = .25

nb_dense = 64


model = Sequential()

model.add(Conv2D(n_conv_filters, conv_kernel_size,
                        padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_fraction))


model.add(Conv2D(n_conv_filters, conv_kernel_size*2,
                        padding='same',))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_fraction))

model.add(Conv2D(n_conv_filters, conv_kernel_size*4,
                        padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_fraction))

model.add(Flatten())
model.add(Dense(2*nb_dense, activation="relu"))
model.add(Dense(nb_dense, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


learning_rate = 0.001

adam = Adam(lr=learning_rate)


model.compile(loss='binary_crossentropy', 
              optimizer=adam,
             )

earlystopping = EarlyStopping(monitor='loss',
                              patience=35,
                              verbose=1,
                              mode='auto' )


goal_batch_size = 64
steps_per_epoch = max(2, training_set_indices.size//goal_batch_size)
batch_size = training_set_indices.size//steps_per_epoch
print("steps_per_epoch: ", steps_per_epoch)
print("batch_size: ", batch_size)
epochs = 100
verbose = 1

Y = targets[HSC_ids].values


history = model.fit_generator(dagan_iterator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=(X[testing_set_indices], Y[testing_set_indices]),
                              verbose=verbose,
                              callbacks=[earlystopping],
                              )


print("best performance: ", min(history.history["val_loss"]))


with mpl.rc_context(rc={"figure.figsize": (10,6)}):

    plt.plot(history.history["val_loss"], label="Validation")
    plt.plot(history.history["loss"], label="Training")


    plt.legend()
    
    plt.xlabel("Epoch")
#     plt.ylabel("Loss\n(avg. binary cross-entropy)")
    plt.ylabel("Loss")

    
    plt.ylim(.45, .65)


class_probs = model.predict_proba(X[testing_set_indices]).flatten()
class_probs


with mpl.rc_context(rc={"figure.figsize": (10,6)}):
    sns.distplot(class_probs[Y[testing_set_indices]==True], color="g", label="true dwarfs")
    sns.distplot(class_probs[Y[testing_set_indices]==False], color="b", label="true non-dwarfs")

    plt.xlabel("p(dwarf | image)")
    plt.ylabel("density (galaxies)")

    plt.xlim(0, .7)
    plt.axvline(Y[training_set_indices].mean(), linestyle="dashed", color="black", label="prior\n(from training set)")
    plt.axvline(.5, linestyle="dotted", color="black", label="50/50")

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )



from sklearn import metrics
from sklearn.metrics import roc_auc_score

with mpl.rc_context(rc={"figure.figsize": (10,6)}):
    fpr, tpr, _ = metrics.roc_curve(Y[testing_set_indices], class_probs)
    roc_auc = roc_auc_score(Y[testing_set_indices], class_probs)

    plt.plot(fpr, tpr, label="DNN (AUC = {:.2})".format(roc_auc))
    plt.plot([0,1], [0,1], linestyle="dashed", color="black", label="random guessing")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="best")


from sklearn import metrics
from sklearn.metrics import average_precision_score
with mpl.rc_context(rc={"figure.figsize": (10,6)}):
    precision, recall, _ = metrics.precision_recall_curve(Y[testing_set_indices], class_probs)
    pr_auc = average_precision_score(Y[testing_set_indices], class_probs)

    plt.plot(recall, precision, label="AUC = {:.2}".format(pr_auc))
    plt.plot([0,1], [Y[testing_set_indices].mean()]*2, linestyle="dashed", color="black")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("PR Curve")

    plt.legend(loc="best")
