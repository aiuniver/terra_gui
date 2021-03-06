from tensorflow import keras

# from terra_ai.training.customcallback import ClassificationCallback, SegmentationCallback, RegressionCallback, \
#     TimeseriesCallback, ObjectdetectionCallback
from terra_ai.custom_objects.customlosses import DiceCoef

custom_losses_dict = {"dice_coef": DiceCoef, "mean_io_u": keras.metrics.MeanIoU}

# task_type_defaults_dict = {
#             "classification": {
#                 "optimizer_name": "Adam",
#                 "loss": "categorical_crossentropy",
#                 "metrics": ["accuracy"],
#                 "batch_size": 32,
#                 "epochs": 20,
#                 "shuffle": True,
#                 "clbck_object": ClassificationCallback,
#                 "callback_kwargs": {
#                     "metrics": ["loss", "accuracy"],
#                     "step": 1,
#                     "class_metrics": [],
#                     "num_classes": 2,
#                     "data_tag": "images",
#                     "show_best": True,
#                     "show_worst": False,
#                     "show_final": True,
#                     "dataset": None,
#                     "exchange": None,
#                 },
#             },
#             "segmentation": {
#                 "optimizer_name": "Adam",
#                 "loss": "categorical_crossentropy",
#                 "metrics": ["dice_coef"],
#                 "batch_size": 16,
#                 "epochs": 20,
#                 "shuffle": True,
#                 "clbck_object": SegmentationCallback,
#                 "callback_kwargs": {
#                     "metrics": ["dice_coef"],
#                     "step": 1,
#                     "class_metrics": [],
#                     "num_classes": 2,
#                     "data_tag": "images",
#                     "show_best": True,
#                     "show_worst": False,
#                     "show_final": True,
#                     "dataset": None,
#                     "exchange": None,
#                 },
#             },
#             "regression": {
#                 "optimizer_name": "Adam",
#                 "loss": "mse",
#                 "metrics": ["mae"],
#                 "batch_size": 32,
#                 "epochs": 20,
#                 "shuffle": True,
#                 "clbck_object": RegressionCallback,
#                 "callback_kwargs": {
#                     "metrics": ["loss", "mse"],
#                     "step": 1,
#                     "plot_scatter": True,
#                     "show_final": True,
#                     "dataset": None,
#                     "exchange": None,
#                 },
#             },
#             "timeseries": {
#                 "optimizer_name": "Adam",
#                 "loss": "mse",
#                 "metrics": ["mae"],
#                 "batch_size": 32,
#                 "epochs": 20,
#                 "shuffle": True,
#                 "clbck_object": TimeseriesCallback,
#                 "callback_kwargs": {
#                     "metrics": ["loss", "mse"],
#                     "step": 1,
#                     "corr_step": 50,
#                     "plot_pred_and_true": True,
#                     "show_final": True,
#                     "dataset": None,
#                     "exchange": None,
#                 },
#             },
#             "object_detection": {
#                 "optimizer_name": "Adam",
#                 "loss": "yolo_loss",
#                 "metrics": [],
#                 "batch_size": 8,
#                 "epochs": 20,
#                 "shuffle": True,
#                 "clbck_object": ObjectdetectionCallback,
#                 "callback_kwargs": {
#                     "metrics": ["loss"],
#                     "step": 1,
#                     "class_metrics": [],
#                     "num_classes": 2,
#                     "data_tag": "images",
#                     "show_best": False,
#                     "show_worst": False,
#                     "show_final": True,
#                     "dataset": None,
#                     "exchange": None,
#                 },
#             },
#         }
