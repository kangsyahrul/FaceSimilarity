import cv2
import numpy as np
import os
from model.DenseNet import *
import matplotlib.pyplot as plt
import tensorflow as tf

def contrastive_loss(logits1, logits2, label, margin, eps=1e-7):
    Dw = tf.sqrt(eps + tf.reduce_sum(tf.square(logits1 - logits2), 1))
    loss = tf.reduce_mean((1. - tf.cast(label, tf.float32))
                          * tf.square(Dw) + tf.cast(label, tf.float32)
                          * tf.square(tf.maximum(margin - Dw, 0)))
    return loss, Dw


# load model
model = DenseNet(**{"k": 32,
        "weight_decay": 0.01,
        "num_outputs": 32,
        "units_per_block": [
                                6,
                                12,
                                24,
                                16
                            ],
        "momentum": 0.997,
        "epsilon": 0.001,
        "initial_pool": True})
checkpoint_directory = "./model"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# TODO: DETERMINE IMAGES PATH HERE
# NOTE: AT LEAST, THERE ARE TWO IMAGES FOR SELFIE AND CARD!
PATHS_SELFIE = ['./faces/syahrul_selfie.jpg',
                './faces/syahrul_selfie_2.jpg',
                './faces/syahrul_selfie_3.jpg',
                './faces/elon_selfie.jpg',
                './faces/elon_selfie_2.jpg',
                './faces/elon_selfie_3.jpg',
                ]
PATHS_CARD = ['./faces/syahrul_ktp.jpg'] * len(PATHS_SELFIE)
LABEL = 0  # must be a similar face

# load images
imgs_selfie = []
imgs_card = []

for i in range(len(PATHS_SELFIE)):
    img_selfie = cv2.imread(PATHS_SELFIE[i])
    img_card = cv2.imread(PATHS_CARD[i])

    # convert BGR to RGB
    img_selfie = cv2.cvtColor(img_selfie, cv2.COLOR_BGR2RGB)
    img_card = cv2.cvtColor(img_card, cv2.COLOR_BGR2RGB)

    imgs_selfie.append(img_selfie)
    imgs_card.append(img_card)

fig, ax = plt.subplots(ncols=len(PATHS_SELFIE), nrows=2, figsize=(16, 4))
fig.subplots_adjust(hspace=0.3, wspace=0.2)
for i in range(len(PATHS_SELFIE)):
    ax[0][i].imshow(imgs_selfie[i])
    ax[0][i].set_title('Selfie')
    ax[0][i].set_axis_off()

    ax[1][i].imshow(imgs_card[i])
    ax[1][i].set_title('Card')
    ax[1][i].set_axis_off()
plt.show()

GX1 = model(np.array(imgs_selfie, dtype=np.float32) / 255, training=False)
GX2 = model(np.array(imgs_card, dtype=np.float32) / 255, training=False)

loss, Dw = contrastive_loss(GX1, GX2, [LABEL], margin=2.)

f, bx = plt.subplots(2, len(PATHS_SELFIE), figsize=(16, 4))
f.subplots_adjust(hspace=0.3, wspace=0.2)
for i in range(len(PATHS_SELFIE)):
    bx[0][i].set_title('Sim: ' + str(Dw[i].numpy()))
    bx[0][i].imshow(imgs_selfie[i])
    bx[0][i].set_axis_off()

    # bx[1][i].set_title("Label: " + str(LABEL))
    bx[1][i].imshow(imgs_card[i])
    bx[1][i].set_axis_off()
plt.show()
