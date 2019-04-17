import matplotlib
# matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




def img_show(file):
    # X_img = np.load("data/images.small.npy")
    # X_img = X_img.transpose([0,2,3,1])

    X_img = np.load(file)
    X_img = X_img.transpose([1,2,0])
    print("\n++ X_img.shape: {0}\n".format(X_img.shape))

    # select image
    plt.imshow(X_img)
    plt.show()



import pandas as pd

HSC_ids = np.load("data/HSC_ids.npy")
HSC_ids


df = pd.read_csv("data/2018_02_23-all_objects.csv")
df = df[df.selected]

df = df.drop_duplicates("HSC_id").set_index("HSC_id").loc[HSC_ids][["photo_z", "log_mass"]]


targets = (df.log_mass > 8) & (df.log_mass < 9) & (df.photo_z < .15)

# print(targets.keys()[0])

file = "43158176442374224-cutout.npy"

img_show("data/npy_files/" + file)

# print(df.keys())
# print(df['photo_z'])
# print(df['photo_z'].keys())
print(df['photo_z'][43158176442374224])
print(df['log_mass'][43158176442374224])







