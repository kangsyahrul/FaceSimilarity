import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getFaces(title, img_bgr, w_desired, h_desired):
    # calculating auto crop based on desire image size
    ratio = w_desired / h_desired
    ratio_w = ratio if ratio > 1 else 1
    ratio_h = 1 if ratio > 1 else ratio

    # resize image
    h, w, d = img_bgr.shape
    if w > h:
        img_bgr = cv2.resize(img_bgr, (720, h * 720 // w))
    else:
        img_bgr = cv2.resize(img_bgr, (w * 720 // h, 720))

    # convert to grayscale
    (img_h, img_w, img_d) = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # detect all faces if any
    faces_raw = face_cascade.detectMultiScale(img_gray, 1.1, 4)
    faces_result = []
    faces_coordinat = []

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
            image = np.copy(img_bgr[y1:y2, x1:x2])
            image = cv2.resize(image, (w_desired, h_desired))
            # cv2.imshow('Face: {}'.format(title), image)
            faces_result.append(image)
            faces_coordinat.append((x1, y1))

        img_bgr = cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if title == 'Selfie' and len(faces_result) == 2:
        # if selfie detect only two foto, return the above one
        index, x_old, y_old = 0, img_w, img_h
        for i in range(len(faces_coordinat)):
            x, y = faces_coordinat[i]
            if y < y_old:
                y_old = y
                index = i
        return [faces_result[index], ]
    return faces_result
