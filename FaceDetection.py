import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getFaces(title, img_bgr, w_desired, h_desired):
    # calculating auto crop based on desire image size
    ratio = w_desired / h_desired
    ratio_w = ratio if ratio > 1 else 1
    ratio_h = 1 if ratio > 1 else ratio

    # convert to grayscale
    (img_h, img_w, img_d) = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # detect all faces if any
    faces_raw = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    faces_result = []

    # Draw the rectangle around each face
    for (x, y, w, h) in faces_raw:
        if w == 0 or h == 0:
            # no face detected
            continue

        # calculating ratio
        cx = x + w//2
        cy = y + h//2

        w_min = 0
        if w < h:
            w_min = w//2
        else:
            w_min = h//2

        dx = int(w_min * ratio_w)
        dy = int(w_min * ratio_h)

        y1, y2 = cy - dy, cy + dy
        x1, x2 = cx - dx, cx + dx

        if not (y1 < 0 or x1 < 0 or y2 > img_h or x2 > img_w):
            image = np.copy(img_bgr[cy - dy:cy + dy, cx - dx:cx + dx])
            image = cv2.resize(image, (w_desired, h_desired))
            cv2.imshow('Face: {}'.format(title), image)
            faces_result.append(image)

        img_bgr = cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return faces_result
