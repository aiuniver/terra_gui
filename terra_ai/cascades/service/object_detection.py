import numpy as np
import torch


def YoloV5(version: str = "small", render_img: bool = True):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5' + version[0].lower(), pretrained=True, force_reload=True)

    def fun(frame: np.ndarray):
        out = model(frame)
        if render_img:
            return out.render()[0]

        return out.xyxy[0].cpu().numpy()

    return fun
