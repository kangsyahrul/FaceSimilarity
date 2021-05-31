#!/usr/bin/env python3

# README: program ini dijalankan sehari sekali untuk melakukan verifikasi kepada semua user yang sedang waiting list
# NOTE: program ini tidak digunakan lagi karena telah menggunakan API dengan Flask, silahkan buka main.py

import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
import FaceDetection as fd
import cv2
from model.DenseNet import *
import numpy as np


# FUNCTIONS
def contrastive_loss(logits1, logits2, label, margin, eps=1e-7):
    Dw = tf.sqrt(eps + tf.reduce_sum(tf.square(logits1 - logits2), 1))
    loss = tf.reduce_mean((1. - tf.cast(label, tf.float32))
                          * tf.square(Dw) + tf.cast(label, tf.float32)
                          * tf.square(tf.maximum(margin - Dw, 0)))
    return loss, Dw

def saveImage(file_path, response):
    file = open(file_path, "wb")
    file.write(response.content)
    file.close()


# CONSTANT VALUE
WIDTH_DESIRED, HEIGHT_DESIRED = 128, 128

# INITIALIZE SERVICE
cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred)

# BUILD 'PHOTOS' DIRECOTY
# check if directory is not exists
if not os.path.exists('photos'):
    os.mkdir('photos')

# delete all photos on the direcory
print('Deleting all photos...')
for f in os.listdir('photos'):
    # chek if it is directory, if yes, iterate through it
    path_0 = os.path.join('photos', f)
    if os.path.isdir(path_0):
        # iterate through directory
        for f1 in os.listdir(path_0):
            # delete file
            path_1 = os.path.join('photos', f, f1)
            os.remove(path_1)
            print('Deleted: {}'.format(path_1))
        # delete directory
        os.rmdir(path_0)
        print('Deleted: {}'.format(path_0))
    else:
        # delete file
        os.remove(path_0)
        print('Deleted: {}'.format(path_0))


# Collect all users id and photos link
print('Downloading photos...')

db = firestore.client()
users_ref = db.collection(u'waiting_list')
docs = users_ref.stream()

users = [] # {uid: ..., link_ktp: .., link_selfie: ..., img_ktp: ..., img_selfie: ...}
for doc in docs:
    val = doc.to_dict()

    # create folder
    path = os.path.join('photos', doc.id)
    os.mkdir(path)

    # download & save image: selfie
    print('Downloading Selfie: ', doc.id)
    file_path = os.path.join(path, 'foto_selfie.jpg')
    saveImage(file_path, requests.get(val['photo_selfie']))
    img_selfie = cv2.imread(file_path)
    face_selfie = fd.getFaces('Selfie', img_selfie, WIDTH_DESIRED, HEIGHT_DESIRED)
    if len(face_selfie) != 1:
        print('WARNING: Could not verifying selfie photo for {}'.format(doc.id))
        print('WARNING MESSAGE: no face or multiple faces detected! Total: ', len(face_selfie))
        continue

    # download & save image: ktp
    print('Downloading KTP: ', doc.id)
    file_path = os.path.join(path, 'foto_ktp.jpg')
    saveImage(file_path, requests.get(val['photo_ktp']))
    img_ktp = cv2.imread(file_path)
    face_ktp = fd.getFaces('KTP', img_ktp, WIDTH_DESIRED, HEIGHT_DESIRED)
    if len(face_ktp) != 1:
        print('WARNING: Could not verifying ktp photo for {}'.format(doc.id))
        print('WARNING MESSAGE: no face or multiple faces detected! Total: ', len(face_ktp))
        continue

    # save faces
    cv2.imwrite(os.path.join(path, 'face_ktp.jpg'), face_ktp[0])
    cv2.imwrite(os.path.join(path, 'face_selfie.jpg'), face_selfie[0])

    # convert BGR to RGB
    face_ktp = cv2.cvtColor(face_ktp[0], cv2.COLOR_BGR2RGB)
    face_selfie = cv2.cvtColor(face_selfie[0], cv2.COLOR_BGR2RGB)
    users.append({'uid': doc.id, 'face_ktp': face_ktp, 'face_selfie': face_selfie, 'link_ktp': val['photo_ktp'],
                  'link_selfie': val['photo_selfie']})


# load the model
print('Loading model...')
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
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

print('Predicting ...')
for user in users:
    # predict
    GX1 = model(np.array([user['face_selfie']], dtype=np.float32) / 255, training=False)
    GX2 = model(np.array([user['face_ktp']], dtype=np.float32) / 255, training=False)

    loss, Dw = contrastive_loss(GX1, GX2, [1], margin=2.)

    print('{}: {:.3f}'.format(user['uid'], Dw[0].numpy()))

    # TODO: UPDATE FIRESTORE
    doc_ref = db.collection(u'users').document(u'{}'.format(user['uid']))
    doc_ref.set({
        u'status': u'verified' if Dw[0].numpy() < 1 else u'rejected',
        u'verification_score': u'{:.3f}'.format(Dw[0].numpy()),
    })
