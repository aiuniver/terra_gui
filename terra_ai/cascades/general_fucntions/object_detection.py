import tensorflow as tf
import numpy as np


def main(**params):
    # classes = params['classes_names']

    def fun(predict):
        print(predict)
        print('\n\n\n', len(predict), '\n\n\n')
        # print('\n\n\n', len(predict), '\n\n\n')
        # print(predict)
        # pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in predict]
        # pred_bbox = tf.concat(pred_bbox, axis=0)
        #
        # bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        # bboxes = nms(bboxes, iou_threshold, method='nms')
        #
        # image = draw_bbox(original_image, bboxes, CLASSES=classes, rectangle_colors=rectangle_colors)
        # # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        #
        # if output_path != '':
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(output_path, image)
        # if show:
        #     # Show the image
        #     cv2.imshow("predicted image", image)
        #     # Load and hold the image
        #     cv2.waitKey(0)
        #     # To close the window after the required kill value was provided
        #     cv2.destroyAllWindows()

        return 'nice'

    return fun
