#!/usr/bin/env python3

# ==================================================================================================== #
# Program ini digunakan untuk API aplikasi android
# Cara menggunakan: silahkan akses link berikut -> https://b21-cap0181.et.r.appspot.com/
# Untuk melakukan verifikasi data diri pada user denga user_id "user_id" silahkan buka link
# https://b21-cap0181.et.r.appspot.com/verify/user_id
# ==================================================================================================== #

# IMPORT
from flask import Flask, request, Response
import json
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
import FaceDetection as fd
import cv2
from model.DenseNet import *
import numpy as np

# INIT FLASK
app = Flask(__name__)


# CONSTANT VALUE
WIDTH_DESIRED, HEIGHT_DESIRED = 128, 128

# INITIALIZE SERVICE
cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred)
db = firestore.client()
FCM_KEY = "AAAAWOAwL2c:APA91bFtSCB-VdaOmBTB2MK74RioueWPyasVefy2XNCk3dJJIc9a7369KLIk-U2XatL5s-xVN2Hlr4PrWExTH1Ghw9dqs1xyR0IwTHWULl8wzTpGuVp6HwCjLg8Jkp4cVrREzRVdvDEn"

# BUILD 'PHOTOS' DIRECTORY if not exists
if not os.path.exists('photos'):
    os.mkdir('photos')

# DELETE ALL PHOTOS
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

# LOAD MODEL
print('Loading model...')
model = DenseNet(**{"k": 32,
        "weight_decay": 0.01,
        "num_outputs": 32,
        "units_per_block": [6, 12, 24, 16],
        "momentum": 0.997,
        "epsilon": 0.001,
        "initial_pool": True})
checkpoint_directory = "./model"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

# INITIALIZE DONE
print('Initialize Done!')


# FUNCTIONS
def contrastive_loss(logits1, logits2, label, margin, eps=1e-7):
    Dw = tf.sqrt(eps + tf.reduce_sum(tf.square(logits1 - logits2), 1))
    loss = tf.reduce_mean((1. - tf.cast(label, tf.float32)) * tf.square(Dw)
                          + tf.cast(label, tf.float32) * tf.square(tf.maximum(margin - Dw, 0)))
    return loss, Dw


def saveImage(file_path, response):
    file = open(file_path, "wb")
    file.write(response.content)
    file.close()


@app.route('/')
def home():
    return 'Selamat Datang di The Folks'


def deletePhotos(user_id):
    _path_0 = os.path.join('photos', user_id)
    if os.path.exists(_path_0):
        # DELETE PHOTOS
        for file in os.listdir(_path_0):
            _path_1 = os.path.join(_path_0, file)
            os.remove(_path_1)
            print('Deleted: ', _path_1)
        os.rmdir(_path_0)
        print('Deleted: ', _path_0)


def verifyFailed(user_id, reason):
    # deletePhotos(user_id)
    js = {'user_id': user_id, 'is_success': False, 'verification_score': -1, 'message': reason}
    return Response(json.dumps(js), mimetype='application/json')


def sendNotif(user_id, title, message):
    # DOWNLOADING TOKEN
    users_ref = db.collection('users').document(user_id)
    user = users_ref.get().to_dict()
    if user is None:
        return 'Failed to send notification. User not found!'

    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        "Authorization": "key=" + FCM_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "to": user['token'],
        "collapse_key": "type_a",
        "notification": {"body": message, "title": title}
    }

    x = requests.post(url, data=json.dumps(data), headers=headers)
    return x.text


@app.route('/notification', methods=['GET'])
def notification():
    user_id = None
    if request.method == 'GET':
        user_id = request.args.get('user_id')

    if user_id is None:
        return 'ERROR: user_id is null'

    return sendNotif(user_id, 'Judul Notifikasi', 'Lorem ipsum dolor sit amet')


@app.route('/verify', methods=['GET'])
def verify():
    if request.method == 'GET':
        user_id = request.args.get('user_id')

        # DOWNLOAD DATABASE
        print('Downloading database...')
        users_ref = db.collection('waiting_list').document(user_id)
        val = users_ref.get().to_dict()
        if val is None:
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, anda belum mengupload foto selfie dan ktp anda')
            return verifyFailed(user_id, 'Photos not found!')
        print('Val: ', val)

        # CREATE FOLDER
        path = os.path.join('photos', user_id)
        if not os.path.exists(path):
            os.mkdir(path)

        # DOWNLOAD IMAGES
        print('Downloading Selfie: ', user_id)
        file_path = os.path.join(path, 'foto_selfie.jpg')
        if 'photo_selfie' not in val:
            print('ERROR: Could not verifying ktp photo for {}'.format(user_id))
            print('ERROR MESSAGE: selfie photo not found!')
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, data diri anda gagal diverifikasi. Anda belum mengupload foto selfie.')
            return verifyFailed(user_id,
                                'Could not verifying ktp photo. Selfie photo not found!')
        saveImage(file_path, requests.get(val['photo_selfie']))
        img_selfie = cv2.imread(file_path)
        face_selfie = fd.getFaces('Selfie', img_selfie, WIDTH_DESIRED, HEIGHT_DESIRED)
        if len(face_selfie) != 1:
            print('ERROR: Could not verifying selfie photo for {}'.format(user_id))
            print('ERROR MESSAGE: no face or multiple faces detected! Total: ', len(face_selfie))
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, data diri anda gagal diverifikasi. Foto selfie anda tidak sesuai. Ditemukan {} wajah pada foto tersebut'.format(len(face_selfie)))
            return verifyFailed(user_id,
                                'Could not verifying selfie photo. No face or multiple faces detected! Total: {}'
                                .format(len(face_selfie)))

        # download & save image: ktp
        print('Downloading KTP: ', user_id)
        file_path = os.path.join(path, 'foto_ktp.jpg')
        if 'photo_ktp' not in val:
            print('ERROR: Could not verifying ktp photo for {}'.format(user_id))
            print('ERROR MESSAGE: selfie photo not found!')
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, data diri anda gagal diverifikasi. Anda belum mengupload foto selfie.')
            return verifyFailed(user_id,
                                'Could not verifying ktp photo. Selfie photo not found!')
        saveImage(file_path, requests.get(val['photo_ktp']))
        img_ktp = cv2.imread(file_path)
        face_ktp = fd.getFaces('KTP', img_ktp, WIDTH_DESIRED, HEIGHT_DESIRED)
        if len(face_ktp) != 1:
            print('ERROR: Could not verifying ktp photo for {}'.format(user_id))
            print('ERROR MESSAGE: no face or multiple faces detected! Total: ', len(face_ktp))
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, data diri anda gagal diverifikasi. Foto KTP anda tidak sesuai. Ditemukan {} wajah pada foto tersebut'.format(
                          len(face_ktp)))
            return verifyFailed(user_id,
                                'Could not verifying ktp photo. No face or multiple faces detected! Total: {}'
                                .format(len(face_ktp)))

        # SAVE FACES
        print('Saving faces...')
        cv2.imwrite(os.path.join(path, 'face_ktp.jpg'), face_ktp[0])
        cv2.imwrite(os.path.join(path, 'face_selfie.jpg'), face_selfie[0])

        # CONVERT BGR to RGB
        face_ktp = cv2.cvtColor(face_ktp[0], cv2.COLOR_BGR2RGB)
        face_selfie = cv2.cvtColor(face_selfie[0], cv2.COLOR_BGR2RGB)

        # PREDICT
        print('Predicting...')
        GX1 = model(np.array([face_selfie], dtype=np.float32) / 255, training=False)
        GX2 = model(np.array([face_ktp], dtype=np.float32) / 255, training=False)
        loss, Dw = contrastive_loss(GX1, GX2, [1], margin=2.)
        js = {'user_id': user_id, 'is_success': True, 'verification_score': str(Dw[0].numpy()),
              'message': 'Predicting similarity success!'}

        # DELETE DIRECTORY
        # print('Deleting {}\'s photos...'.format(user_id))
        # deletePhotos(user_id)

        # UPDATE FIRESTORE
        print('Updating firestore...')
        db.collection(u'users').document(u'{}'.format(user_id)).update({
            u'status': u'verified' if Dw[0].numpy() < 1 else u'rejected',
            u'verification_score': u'{:.3f}'.format(Dw[0].numpy()),
        })

        # SEND NOTIFICATION
        if Dw[0].numpy() < 1:
            sendNotif(user_id, 'Verifikasi Berhasil',
                      'Selamat, data diri anda berhasil diverifikasi. Hasil verifikasi: {:.3f}'.format(Dw[0].numpy()))
        else:
            sendNotif(user_id, 'Verifikasi Gagal',
                      'Maaf, data diri anda gagal diverifikasi. Hasil verifikasi: {:.3f}'.format(Dw[0].numpy()))

        # return json
        print('Result: ', js)
        return Response(json.dumps(js), mimetype='application/json')
    else:
        return 'Please use METHOD GET INSTEAD.'


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
