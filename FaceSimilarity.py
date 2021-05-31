#!/usr/bin/env python3

# README: program ini digunakan untuk mendapatkan nilai prediksi dari face similarity



# Collect all users id and photos link
print('Downloading photos...')



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
