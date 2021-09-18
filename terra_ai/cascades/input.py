import cv2


def video(path):

    cap = cv2.VideoCapture(path)

    while cap.isOpened:

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        yield frame
