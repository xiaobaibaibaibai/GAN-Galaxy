from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd

import seaborn as sns
sns.set(font_scale=2, style="ticks")

HSC_ids = np.load("data/HSC_ids.npy")
HSC_ids

X = np.load("data/images.small.npy")
X.shape


# Get targets
df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]
df.head()



targets = df.drop_duplicates("HSC_id") \
            .set_index("HSC_id")
    


targets = (targets.log_mass > 8) & (targets.log_mass < 9) & (targets.photo_z < .15)
print(targets.mean())
print(targets.sum())


# Split training and validation sets
batch_size = 64

np.random.seed(seed=0)

randomized_indices = np.arange(X.shape[0])
np.random.shuffle(randomized_indices)

testing_fraction = 0.2
# make sure testing set size is an even multiple of 64
num_testing = (int(testing_fraction*X.shape[0]) // batch_size) * batch_size

testing_set_indices = randomized_indices[:int(num_testing)]
training_set_indices = np.array(list(set([*randomized_indices]) - set([*testing_set_indices])))

testing_set_indices.size
training_set_indices.size


# Setup standard augmentation
from keras.preprocessing.image import ImageDataGenerator

print('Using real-time _simple_ data augmentation.')

h_before, w_before = X[0,0].shape
print("image shape before: ({},{})".format(h_before, w_before))

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    horizontal_flip=True, # randomly apply a reflection (in x)
    vertical_flip=True, # randomly apply a reflection (in y)
    rotation_range=0, # randomly apply a rotation of angle randomly between 0 and `rotation_range`
    zoom_range=0.0,
    shear_range=0.0,
    channel_shift_range=0.0,
    rescale=0,
    width_shift_range=0.002,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.002,  # randomly shift images vertically (fraction of total height)
)

datagen.fit(X[training_set_indices])

# Setup Classifier
from classifier import Classifier

input_shape = X.shape[1:]

classifier_model = Classifier(input_shape)
classifier_model.configure_optimizer(lr=0.001)
classifier_model.build_model()
classifier_model.configure_early_stopping()

Y = targets[HSC_ids].values

data_iterator = datagen.flow(X[training_set_indices],
                             Y[training_set_indices],
                             batch_size=classifier_model.batch_size,
                            )


# Run Basic Classifier
history = classifier_model.fit_model(X, Y, 
                                     training_set_indices,
                                     testing_set_indices,
                                     data_iterator,
                                    )

# Check Classifier Performance
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
    
    plt.ylim(.4, .7)
    
    # Force only integer labels, not fractional labels
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # save figure
    plt.savefig("f0.png")


class_probs = classifier_model.model \
                              .predict_proba(X[testing_set_indices]) \
                              .flatten()

print("predict probability: ", class_probs)
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

    plt.savefig("f1.png")



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

    plt.savefig("f2.png")




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

    plt.savefig("f3.png")


















