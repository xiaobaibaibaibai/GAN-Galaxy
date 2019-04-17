import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import matplotlib as mpl

import numpy as np
import pandas as pd
import os

import tensorflow as tf
from keras import backend as K

import seaborn as sns
sns.set(font_scale=2, style="ticks")

# my code
import misc
import gan


HSC_ids = np.load("data/HSC_ids.npy")
HSC_ids

X = np.load("data/images.small.npy")
X.shape

X_img = X.copy().transpose([0,2,3,1])
X_img.shape

image_size = X.shape[-1]
image_shape = X.shape[1:]
image_size



df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]

# df = df.drop_duplicates("HSC_id").set_index("HSC_id").loc[HSC_ids][["photo_z", "log_mass", "rcmodel_mag", "icmodel_mag"]]
df = df.drop_duplicates("HSC_id").set_index("HSC_id").loc[HSC_ids][["photo_z", "log_mass"]]


targets = (df.log_mass > 8) & (df.log_mass < 9) & (df.photo_z < .15)

y_conditionals = df.values

# y_conditionals_for_visualization = np.array([.14, 8.51, 21.15, 23.71])
y_conditionals_for_visualization = np.array([.14, 8.51])

# values copied from output of `simple gan.ipynb`
# standardizer = misc.Standardizer(means = np.array([0.21093612, 8.62739865, 21.54070648, 21.28554743]),
#                                  std = np.array([0.30696933, 0.63783586, 1.13836541, 1.14504214]))
standardizer = misc.Standardizer(means = np.array([0.21093612, 8.62739865]),
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
                                  "GAN-GALAXY",
                                  "models/classify_DAGAN/checkpoints")
else:
    num_epochs = 1
    # use a dir inside the repo
    checkpoint_dir = "models/ri_gan/checkpoints"

# batch_size = 64 # set above
z_dim = 100
dataset_name = "galaxy"
result_dir = "models/classify_DAGAN"
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
plt.savefig("result.png")

from classifier import Classifier

input_shape = X.shape[1:]

classifier_model = Classifier(input_shape)
classifier_model.configure_optimizer(lr=0.001)
classifier_model.build_model()
classifier_model.configure_early_stopping()


Y = targets[HSC_ids].values


history = classifier_model.fit_model(X, Y, 
                                     training_set_indices,
                                     testing_set_indices,
                                     dagan_iterator,
                                    )

from sklearn.metrics import log_loss
p = Y[training_set_indices].mean()
prior_loss = log_loss(Y[testing_set_indices], 
                      [p]*testing_set_indices.size)

print("performance (prior): {:.3f}".format(prior_loss))
print("performance (best):  {:.3f}".format(min(history.history["val_loss"])))


from matplotlib.ticker import MaxNLocator

with mpl.rc_context(rc={"figure.figsize": (10,6)}):

    plt.plot(history.history["val_loss"], label="Validation")
    plt.plot(history.history["loss"], label="Training")
    
    plt.axhline(prior_loss, label="Prior", 
                linestyle="dashed", color="black")

    plt.legend(loc="best")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss\n(mean binary cross-entropy)")
    
#     plt.ylim(.45, .65)
    
    # Force only integer labels, not fractional labels
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))



class_probs = classifier_model.model \
                              .predict_proba(X[testing_set_indices]) \
                              .flatten()
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