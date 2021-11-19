import numpy as np
import torch


def yolo_v5(version: str = "small"):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5' + version[0].lower(), pretrained=True)

    def fun(frame: np.ndarray):
        out = model(frame)
        out = out.render()[0]
        return out

    return fun
