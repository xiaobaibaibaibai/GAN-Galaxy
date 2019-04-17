import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

X_img = np.load("data/images.small.npy")

X_img = X_img.transpose([0,2,3,1])
print("\n++ X_img.shape: {0}\n".format(X_img.shape))

# select image
plt.imshow(X_img[1])
plt.show()
