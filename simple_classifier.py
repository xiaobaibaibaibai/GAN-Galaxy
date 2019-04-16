from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd

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

HSC_ids = np.load("data/HSC_ids.npy")
HSC_ids

X = np.load("data/images.small.npy")
X.shape

# Get targets
df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]
df.head()

targets = df.drop_duplicates("HSC_id") \
            .set_index("HSC_id") \

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

# Setup keras model
n_conv_filters = 16
conv_kernel_size = 4
input_shape = X.shape[1:]

dropout_fraction = .25

nb_dense = 64

input_shape


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
decay = 1e-5
momentum = 0.9

sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

adam = Adam(lr=learning_rate)

model.compile(loss='binary_crossentropy', 
#               optimizer=sgd, 
              optimizer=adam,
#               metrics=["accuracy"]
             )

earlystopping = EarlyStopping(monitor='loss',
                              patience=35,
                              verbose=1,
                              mode='auto' )

# Run Basic Keras Model

goal_batch_size = 64
steps_per_epoch = max(2, training_set_indices.size//goal_batch_size)
batch_size = training_set_indices.size//steps_per_epoch
print("steps_per_epoch: ", steps_per_epoch)
print("batch_size: ", batch_size)
epochs = 100
verbose=1

Y = targets[HSC_ids].values

# %%timeit -r 1 -n 1
history = model.fit_generator(datagen.flow(X[training_set_indices], Y[training_set_indices],
                                           batch_size=batch_size,
                                          ),
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

    plt.savefig("Classifier_Baseline0.png")




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

    plt.savefig("Classifier_Baseline1.png")





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

    plt.savefig("ROC_Curve.png")


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

    plt.savefig("PR_Curve.png")













