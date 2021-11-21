import numpy as npй
import cv2
from PIL import Image, ImageDraw
from random import randrange


def plot_tracks():
    colors = {}

    def fun(bboxes, img):
        # Plots one bounding box on image img
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        for x in bboxes:
            id = int(x[-1])
            if not (id in colors.keys()):
                colors[id] = [randrange(1, 256) for i in range(3)]
            color = colors[id]

            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            t_size = (114, 16)
            c2 = c1[0] + t_size[0] + 5, c1[1] - t_size[1] - 5
            c1 = c1[0], c1[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil_img)
            d.text((c1[0] + 5, c1[1] - 17), f"ID - {id}", fill=(225, 255, 255, 0))

            img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        return img
    return fun
