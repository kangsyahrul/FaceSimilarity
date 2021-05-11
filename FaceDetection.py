import cv2
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)

# desired_w, desired_h = 400, 400
desired_w, desired_h = 360, 480
ratio = desired_w/desired_h

if ratio > 1:
    # landscape
    ratio_w = ratio
    ratio_h = 1
else:
    # potrait
    ratio_w = 1
    ratio_h = 1/ratio

while True:
    start_time = time.time()
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    (img_h, img_w, d) = img.shape
    img_w, img_h = img_w // 2, img_h // 2

    img = cv2.resize(img, (img_w, img_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        if w == 0 or h == 0:
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
            image = img[cy - dy:cy + dy, cx - dx:cx + dx]
            image = cv2.resize(image, (desired_w, desired_h))
            cv2.imshow('Face', image)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    fps = 1/(time.time() - start_time)
    img = cv2.putText(img, 'FPS: {:.02f} '.format(fps), (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
