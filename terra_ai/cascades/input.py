import cv2


def video(path):
    while True:
        cap = cv2.VideoCapture(path)

        while cap.isOpened:

            ret, frame = cap.read()

            yield frame
