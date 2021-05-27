import cv2
import time
import FaceDetection as fd

# To capture video from webcam.
cap = cv2.VideoCapture(0)

WIDTH_DESIRED, HEIGHT_DESIRED = 128, 128

while True:
    # Read the frame
    _, img = cap.read()
    start_time = time.time()

    # Convert to grayscale
    (img_h, img_w, d) = img.shape
    img_w, img_h = img_w // 2, img_h // 2

    faces = fd.getFaces('Selfie', img, WIDTH_DESIRED, HEIGHT_DESIRED)
    if len(faces) > 0:
        cv2.imshow('Face', faces[0])

    # Display
    fps = 1/(time.time() - start_time)
    img = cv2.putText(img, 'FPS: {:.02f} '.format(fps), (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Web Cam', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
