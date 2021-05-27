import cv2
import time
import numpy as np
import FaceDetection as fd

# TODO: DETERMINE INPUT IMAGES
PATH_SELFIE = './image/elon_selfie_3.jpg'
# PATH_SELFIE = './image/syahrul_selfie_3.jpg'
PATH_CARD = './image/syahrul_ktp.jpg'

# determine image resolution for NN input
WIDTH_DESIRED, HEIGHT_DESIRED = 128, 128

# load images
img_selfie = cv2.imread(PATH_SELFIE)
h, w, d = img_selfie.shape
if w > h:
    img_selfie = cv2.resize(img_selfie, (720, h * 720 // w))
else:
    img_selfie = cv2.resize(img_selfie, (w * 720 // h, 720))

img_card = cv2.imread(PATH_CARD)
h, w, d = img_card.shape
if w > h:
    img_card = cv2.resize(img_card, (720, h * 720 // w))
else:
    img_card = cv2.resize(img_card, (w * 720 // h, 720))

# detect face(s) on selfie image
faces = fd.getFaces('Selfie', img_selfie, WIDTH_DESIRED, HEIGHT_DESIRED)
cv2.imshow('Selfie', img_selfie)
if len(faces) == 1:
    # export image
    PATH_SELFIE_EXPORT = './faces/{}'.format(PATH_SELFIE.split('/')[-1])
    cv2.imwrite(PATH_SELFIE_EXPORT, faces[0])
    print('Face on Selfie exported to {}'.format(PATH_SELFIE_EXPORT))

else:
    # error
    print('DETECTION ON SELFIE ERROR, FACE(s) FOUND: {}'.format(len(faces)))

# detect face(s) on card image
faces = fd.getFaces('Card', img_card, WIDTH_DESIRED, HEIGHT_DESIRED)
cv2.imshow('Card', img_card)
if len(faces) == 1:
    # export image
    PATH_CARD_EXPORT = './faces/{}'.format(PATH_CARD.split('/')[-1])
    cv2.imwrite(PATH_CARD_EXPORT, faces[0])
    print('Face on Card exported to {}'.format(PATH_CARD_EXPORT))

else:
    # error
    print('DETECTION ON CARD ERROR, FACE(s) FOUND: {}'.format(len(faces)))


cv2.waitKey(0)
cv2.destroyAllWindows()
