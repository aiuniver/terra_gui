import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import types
from terra_ai.guiexchange import Exchange
from terra_ai.trds import DTS

__version__ = 0.60


class ClassificationCallback(keras.callbacks.Callback):
    """ Callback for classification """

    def __init__(
        self,
        metrics=[],
        step=1,
        class_metrics=[],
        data_tag="images",
        show_worst=False,
        show_best=False,
        show_final=True,
        dataset=DTS(),
        exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли справа отдельно экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли справа отдельно экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
        Returns:
            None
        """
        super().__init__()
        self.step = step
        self.clbck_metrics = metrics
        self.data_tag = data_tag
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.epoch = 0
        self.history = {}
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.classes = 0
        self.predict_cls = {}
        self.idx = 0

        self.classes = self.dataset.num_classes  # количество классов
        self.acls_lst = [
            [[] for i in range(self.classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]
        self.predict_cls = (
            {}
        )  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        pass

    def plot_result(self):
        """
        Returns:
            None:
        """
        plot_data = {}
        msg_epoch = f"Epoch №{self.epoch + 1:03d}"
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.__name__
                val_metric_name = f"val_{metric_name}"
                # определяем, что демонстрируем во 2м и 3м окне
                metric_title = f"{metric_name} and {val_metric_name} {msg_epoch}"
                xlabel = "epoch"
                ylabel = f"{metric_name}"
                labels = (metric_title, xlabel, ylabel)
                plot_data[labels] = [[list(range(len(self.history[metric_name]))),
                                      self.history[metric_name],
                                      f'{metric_name}'],
                                     [list(range(len(self.history[val_metric_name]))),
                                      self.history[val_metric_name],
                                      f'{val_metric_name}']]

            if self.class_metrics:
                for metric_name in self.clbck_metrics:
                    if not isinstance(metric_name, str):
                        metric_name = metric_name.__name__
                    val_metric_name = f"val_{metric_name}"
                    if metric_name in self.class_metrics:
                        classes_title = f"{val_metric_name} of {self.classes} classes. {msg_epoch}"
                        xlabel = "epoch"
                        ylabel = val_metric_name
                        labels = (classes_title, xlabel, ylabel)
                        plot_data[labels] = [[list(range(len(self.predict_cls[val_metric_name][self.idx][j]))),
                                              self.predict_cls[val_metric_name][self.idx][j],
                                              f"{val_metric_name} class {j}", ] for j in range(self.classes)]
            self.Exch.show_plot_data(plot_data)
        pass

    def image_indices(self, count=5) -> np.ndarray:
        """
        Computes indices of images based on instance mode ('worst', 'best')
        Returns: array of best or worst predictions indices
        """
        classes = np.argmax(self.y_true, axis=-1)
        probs = np.array([pred[classes[i]]
                          for i, pred in enumerate(self.y_pred)])
        sorted_args = np.argsort(probs)
        if self.show_best:
            indices = sorted_args[-count:]
        else:
            indices = sorted_args[:count]
        return indices

    def plot_images(self):
        """
        Plot images based on indices in dataset
        Returns: None
        """
        img_indices = self.image_indices()

        classes_labels = np.arange(self.classes)
        data = []
        for idx in img_indices:
            image = self.dataset.x_Val[idx]
            true_idx = np.argmax(self.y_true[idx])
            pred_idx = np.argmax(self.y_pred[idx])
            title = f"Predicted: {classes_labels[pred_idx]} \n Actual: {classes_labels[true_idx]}"
            data.append((image, title))
        self.Exch.show_image_data(data)

    # Распознаём тестовую выборку и выводим результаты
    def recognize_classes(self, y_pred, y_true):
        y_pred_classes = np.argmax(y_pred, axis=-1)
        y_true_classes = np.argmax(y_true, axis=-1)
        classes_accuracy = []
        for j in range(self.classes + 1):
            accuracy_value = 0
            y_true_count_sum = 0
            y_pred_count_sum = 0
            for i in range(y_true.shape[0]):
                y_true_diff = y_true_classes[i] - j
                if not y_true_diff:
                    y_pred_count_sum += 1
                y_pred_diff = y_pred_classes[i] - j
                if not (y_true_diff and y_pred_diff):
                    y_true_count_sum += 1
                if not y_pred_count_sum:
                    accuracy_value = 0
                else:
                    accuracy_value = y_true_count_sum / y_pred_count_sum
            classes_accuracy.append(accuracy_value)
        return classes_accuracy

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.Exch.show_current_epoch(epoch)
        pass

    def on_epoch_end(self, epoch, logs={}):
        """
        Returns:
            {}:
        """
        max_accuracy_value = 0
        self.idx = 0
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for metric_idx in range(len(self.clbck_metrics)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].__name__
                self.clbck_metrics[metric_idx] = metric_name
            else:
                metric_name = self.clbck_metrics[metric_idx]
            val_metric_name = f"val_{metric_name}"

            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > max_accuracy_value:
                max_accuracy_value = logs[val_metric_name]
                self.idx = metric_idx
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {str(metric_name): self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

            if len(self.dataset.x_Val):
                # получаем y_pred и y_true
                y_pred = self.model.predict(self.dataset.x_Val)
                y_true = self.dataset.y_Val

                self.y_true = y_true
                self.y_pred = y_pred

                # распознаем и выводим результат по классам
                classes_accuracy = self.recognize_classes(y_pred, y_true)
                # собираем в словарь по метрикам и классам
                dclsup = {}
                for j in range(self.classes + 1):
                    self.acls_lst[metric_idx][j].append(classes_accuracy[j])
                dcls = {str(val_metric_name): self.acls_lst}
                dclsup.update(dcls)
                self.predict_cls.update(dclsup)

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                self.plot_result()

        self.Exch.print_epoch_monitor(
            f"Epoch {epoch:03d}{epoch_metric_data}{epoch_val_metric_data}"
        )
        return self.predict_cls

    def on_train_end(self, logs={}):
        if self.show_final:
            self.plot_result()
            if self.show_best or self.show_worst:
                self.plot_images()


class SegmentationCallback(keras.callbacks.Callback):
    """ Callback for classification"""

    def __init__(
        self,
        metrics=[],
        step=1,
        class_metrics=[],
        data_tag="images",
        show_worst=False,
        show_best=False,
        show_final=True,
        dataset=DTS(),
        exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик (по умолчанию clbck_metrics = list()), что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: class_metrics (по умолчанию 'best', если  = [], то не будет справа выводить)
            step int(list):             шаг вывода хода обучения, по умолчанию step = 1
            show_worst bool():          выводить ли справа отдельно, плохие метрики, по умолчанию False
            show_final bool ():         выводить ли в конце обучения график, по умолчанию True
            dataset (trds.DTS):         instance of DTS class
            exchange:                   экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        super().__init__()
        self.step = step
        self.clbck_metrics = metrics
        self.data_tag = data_tag
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.epoch = 0
        self.history = {}
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.idx = 0
        self.num_classes = self.dataset.num_classes  # количество классов
        self.acls_lst = [[[] for i in range(self.num_classes + 1)] for i in range(len(self.clbck_metrics))]  #
        self.predict_cls = {}  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        pass

    def plot_result(self) -> None:
        """
        Returns:
            None:
        """
        plot_data = {}
        msg_epoch = f'Epoch №{self.epoch + 1:03d}'
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.__name__
                val_metric_name = f"val_{metric_name}"
                # определяем, что демонстрируем во 2м и 3м окне
                metric_title = f'{metric_name} and {val_metric_name} {msg_epoch}'
                xlabel = "epoch"
                ylabel = f"{metric_name}"
                labels = (metric_title, xlabel, ylabel)
                plot_data[labels] = [[list(range(len(self.history[metric_name]))),
                                      self.history[metric_name],
                                      f'{metric_name}'],
                                     [list(range(len(self.history[val_metric_name]))),
                                      self.history[val_metric_name],
                                      f'{val_metric_name}']]

            if self.class_metrics:
                for idx, metric_name in enumerate(self.clbck_metrics):
                    if not isinstance(metric_name, str):
                        metric_name = metric_name.__name__
                    val_metric_name = f'val_{metric_name}'
                    if metric_name in self.class_metrics:
                        classes_title = f'{val_metric_name} of {self.num_classes} classes. {msg_epoch}'
                        xlabel = 'epoch'
                        ylabel = val_metric_name
                        labels = (classes_title, xlabel, ylabel)
                        plot_data[labels] = [
                            [list(range(len(self.predict_cls[val_metric_name][idx][j]))),
                             self.predict_cls[val_metric_name][idx][j],
                             f"{val_metric_name} class {self.dataset.classes_names[j]}"]
                            for j in range(self.num_classes)]
            self.Exch.show_plot_data(plot_data)
        pass

    def _get_colored_mask(self, mask):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        def index2color(pix, num_classes, classes_colors):
            index = np.argmax(pix)
            color = []
            for i in range(num_classes):
                if index == i:
                    color = classes_colors[i]
            return color

        colored_mask = []
        mask = mask.reshape(-1, self.dataset.num_classes)
        for pix in range(len(mask)):
            colored_mask.append(
                index2color(
                    mask[pix], self.dataset.num_classes, self.dataset.classes_colors
                )
            )
        colored_mask = np.array(colored_mask)
        self.colored_mask = colored_mask.reshape(self.dataset.input_shape)

    def _dice_coef(self, smooth=1.):
        """
        Compute dice coefficient for each mask

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """

        intersection = np.sum(self.y_true * self.y_pred, axis=(1, 2, 3))
        union = np.sum(self.y_true, axis=(1, 2, 3)) + np.sum(self.y_pred, axis=(1, 2, 3))
        self.dice = (2. * intersection + smooth) / (union + smooth)

    def plot_images(self):
        """
        Returns:
            None:
        """

        image_data = []
        true_mask_data = []
        pred_mask_data = []

        self._dice_coef()

        # выбираем 5 лучших либо 5 худших результатов сегментации
        if self.show_best:
            indexes = np.argsort(self.dice)[-5:]
        elif self.show_worst:
            indexes = np.argsort(self.dice)[:5]

        for idx in indexes:
            # исходное изобаржение
            image = np.squeeze(
                self.dataset.x_Val[idx].reshape(self.dataset.input_shape)
            )
            title = "Image"
            image_data.append((image, title))

            # истинная маска
            self._get_colored_mask(self.y_true[idx])
            image = np.squeeze(self.colored_mask)
            title = "Ground truth mask"
            true_mask_data.append((image, title))

            # предсказанная маска
            self._get_colored_mask(self.y_pred[idx])
            image = np.squeeze(self.colored_mask)
            title = "Predicted mask"
            pred_mask_data.append((image, title))

        data = image_data + true_mask_data + pred_mask_data
        self.Exch.show_image_data(data)

    # Распознаём тестовую выборку и выводим результаты
    def evaluate_accuray(self, smooth=1.):
        """
        Compute accuracy for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """
        predsegments = np.argmax(self.y_pred, axis=-1)
        truesegments = np.argmax(self.y_true, axis=-1)
        self.metric_classes = []
        for j in range(self.num_classes):
            summ_val = 0
            for i in range(self.y_true.shape[0]):
                # делаем сегметн класса для сверки
                testsegment = np.ones_like(predsegments[0]) * j
                truezero = np.abs(truesegments[i] - testsegment)
                predzero = np.abs(predsegments[i] - testsegment)
                summ_val += (
                    testsegment.size - np.count_nonzero(truezero + predzero)
                ) / (testsegment.size - np.count_nonzero(predzero) + smooth)
            acc_val = summ_val / self.y_true.shape[0]
            self.metric_classes.append(acc_val)

    def evaluate_dice_coef(self, smooth=1.):
        """
        Compute dice coefficient for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """
        if self.dataset.tags[0] == 'images':
            axis = (1, 2)
        elif self.dataset.tags[0] == 'text':
            axis = 1
        intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        dice = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        self.metric_classes = dice

    def evaluate_loss(self):
        """
        Compute loss for classes

        Returns:
        -------
        None
        """
        self.metric_classes = []
        bce = BinaryCrossentropy()
        for i in range(self.num_classes):
            loss = bce(self.y_true[..., i], self.y_pred[..., i]).numpy()
            self.metric_classes.append(loss)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.Exch.show_current_epoch(epoch)
        pass

    def on_epoch_end(self, epoch, logs={}):
        """
        Returns:
            {}:
        """
        max_accuracy_value = 0
        self.epoch = epoch
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        self.idx = 0
        for metric_idx in range(len(self.clbck_metrics)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].__name__
                self.clbck_metrics[metric_idx] = metric_name
            else:
                metric_name = self.clbck_metrics[metric_idx]
            val_metric_name = f"val_{metric_name}"

            if logs[val_metric_name] > max_accuracy_value:
                max_accuracy_value = logs[val_metric_name]
                self.idx = metric_idx
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {str(metric_name): self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

            # получаем y_pred и y_true
            self.y_pred = self.model.predict(self.dataset.x_Val)
            self.y_true = self.dataset.y_Val

            # вычисляем результат по классам
            if metric_name == 'accuracy':
                self.evaluate_accuray()
            elif metric_name == 'dice_coef':
                self.evaluate_dice_coef()
            elif metric_name == 'loss':
                self.evaluate_loss()

            # собираем в словарь по метрикам и классам
            dclsup = {}
            for j in range(self.num_classes):
                self.acls_lst[metric_idx][j].append(self.metric_classes[j])
            dcls = {val_metric_name: self.acls_lst}
            dclsup.update(dcls)
            self.predict_cls.update(dclsup)

        if self.step > 0:
            if self.epoch % self.step == 0:
                self.plot_result()

        self.Exch.print_epoch_monitor(
            f"Epoch {epoch:03d}{epoch_metric_data}{epoch_val_metric_data}"
        )

    def on_train_end(self, logs={}):
        if self.show_final:
            self.plot_result()
            if self.show_best or self.show_worst:
                self.plot_images()
        pass


class TimeseriesCallback(keras.callbacks.Callback):
    def __init__(
        self,
        metrics=["loss"],
        step=1,
        corr_step=50,
        show_final=True,
        plot_true_and_pred=True,
        dataset=DTS(),
        exchange=Exchange(),
    ):
        """
        Init for timeseries callback
        Args:
            metrics (list):             список используемых метрик (по умолчанию clbck_metrics = list()), что соответсвует 'loss'
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            plot_true_and_pred (bool):  выводить ли графики реальных и предсказанных рядов
            dataset (DTS):              экземпляр класса DTS
            corr_step (int):            количество шагов для отображения корреляции (при <= 0 не отображается)
        Returns:
            None
        """
        super().__init__()
        self.metrics = metrics
        self.step = step
        self.show_final = show_final
        self.plot_true_and_pred = plot_true_and_pred
        self.dataset = dataset
        self.corr_step = corr_step
        self.Exch = exchange

    def plot_result(self):
        showmet = self.losses[self.idx]
        vshowmet = f"val_{showmet}"
        epochcomment = f" epoch {self.epoch + 1}"
        loss_len = len(self.history["loss"])
        data = {}

        loss_title = (f"loss and val_loss {epochcomment}", "epochs", f"{showmet}")
        data.update(
            {
                loss_title: [
                    [range(loss_len), self.history["loss"], "loss"],
                    [range(loss_len), self.history["val_loss"], "val_loss"],
                ]
            }
        )

        metric_title = (
            f"{showmet} metric = {showmet} and {vshowmet}{epochcomment}",
            "epochs",
            f"{showmet}",
        )
        data.update(
            {
                metric_title: [
                    [range(loss_len), self.history[showmet], showmet],
                    [range(loss_len), self.history[vshowmet], vshowmet],
                ]
            }
        )

        if self.plot_true_and_pred:
            y_true, y_pred = self.predicts[vshowmet]
            pred_title = ("Predictions", "epochs", f"{showmet}")
            data.update(
                {
                    pred_title: [
                        [range(len(y_true)), y_true, "Actual"],
                        [range(len(y_pred)), y_pred, "Prediction"],
                    ]
                }
            )
        self.Exch.show_plot_data(data)

    @staticmethod
    def autocorr(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()
        corr = 1
        if (sa > 0) & (sb > 0):
            corr = (mab - ma * mb) / (sa * sb)
        return corr

    @staticmethod
    def collect_correlation_data(y_pred, y_true, channel, corr_steps=10):
        corr_list = []
        autocorr_list = []
        yLen = y_true.shape[0]
        for i in range(corr_steps):
            corr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_pred[i:, channel]
                )
            )
            autocorr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_true[i:, channel]
                )
            )
        corr_label = f"Предсказание на {channel + 1} шаг"
        autocorr_label = "Эталон"
        title = "Автокорреляция"
        correlation_data = {
            title: [
                [range(corr_steps), corr_list, corr_label],
                [range(corr_steps), autocorr_list, autocorr_label],
            ]
        }
        return correlation_data

    def on_train_begin(self, logs=None):
        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]
        self.history = {}
        if len(self.dataset.x_Val):
            self.predicts = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.Exch.show_current_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            else:
                metric_name = self.losses[i]
            val_metric_name = f"val_{metric_name}"
            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            if len(self.dataset.x_Val):
                y_pred = self.model.predict(self.dataset.x_Val)
                y_true = self.dataset.y_Val
                self.predicts[val_metric_name] = (y_true, y_pred)
                self.vmet_name = val_metric_name

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                self.comment = f" epoch {epoch + 1}"
                self.idx = 0
                self.plot_result()
        self.exchange.print_epoch_monitor(
            f"Epoch {epoch:03d}{epoch_metric_data}{epoch_val_metric_data}"
        )

    def on_train_end(self, logs=None):
        if self.show_final:
            self.comment = f"on {self.epoch + 1} epochs"
            self.idx = 0
            self.plot_result()
        if self.corr_step > 0:
            y_true, y_pred = self.predicts[self.vmet_name]
            corr_data = TimeseriesCallback.collect_correlation_data(
                y_pred, y_true, 0, self.corr_step
            )
            # Plot correlation and autocorrelation graphics
            self.exchange.show_plot_data(corr_data)


class RegressionCallback(keras.callbacks.Callback):
    def __init__(
        self,
        metrics,
        step=1,
        data_tag="text",
        show_final=True,
        plot_scatter=False,
        dataset=DTS(),
        exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):         список используемых метрик
            step int():             шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):      выводить ли в конце обучения график, по умолчанию True
            exchange:               экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        super().__init__()
        self.step = step
        self.metrics = metrics
        self.show_final = show_final
        self.plot_scatter = plot_scatter
        self.data_tag = data_tag
        self.dataset = dataset
        self.exchange = exchange
        pass

    def plot_result(self):
        showmet = self.losses[self.idx]
        vshowmet = f"val_{showmet}"
        epochcomment = f" epoch {self.epoch + 1}"
        loss_len = len(self.history["loss"])
        data = {}

        loss_title = f"loss and val_loss{epochcomment}"
        xlabel = "epoch"
        ylabel = "loss"
        key = (loss_title, xlabel, ylabel)
        value = [
            [range(loss_len), self.history["loss"], "loss"],
            [range(loss_len), self.history["val_loss"], "val_loss"],
        ]
        data.update({key: value})

        metric_title = f"{showmet} metric = {showmet} and {vshowmet}{epochcomment}"
        xlabel = "epoch"
        ylabel = f"{showmet}"
        key = (metric_title, xlabel, ylabel)
        value = [
            (range(loss_len), self.history[showmet], showmet),
            (range(loss_len), self.history[vshowmet], vshowmet),
        ]
        data.update({key: value})
        self.exchange.show_plot_data(data)

        if self.plot_scatter:
            data = {}
            scatter_title = "Scatter"
            xlabel = "True values"
            ylabel = "Predictions"
            y_true, y_pred = self.predicts[vshowmet]
            key = (scatter_title, xlabel, ylabel)
            value = [(y_true, y_pred, "Regression")]
            data.update({key: value})
            self.exchange.show_scatter_data(data)

        pass

    def on_train_begin(self, logs={}):
        self.losses = self.metrics + ["loss"]
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]
        self.history = {}
        self.predicts = {}
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.exchange.show_current_epoch(epoch)
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            else:
                metric_name = self.losses[i]
            val_metric_name = f"val_{metric_name}"
            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            if len(self.dataset.x_Val):
                y_pred = self.model.predict(self.dataset.x_Val)
                y_true = self.dataset.y_Val
                self.predicts[val_metric_name] = (y_true, y_pred)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

        if self.step > 0:
            if self.epoch % self.step == 0:
                self.comment = f" epoch {epoch + 1}"
                self.idx = 0
                self.plot_result()

        self.exchange.print_epoch_monitor(
            f"Epoch {epoch:03d}{epoch_metric_data}{epoch_val_metric_data}"
        )
        pass

    def on_train_end(self, logs={}):
        if self.show_final:
            self.comment = f"on {self.epoch + 1} epochs"
            self.idx = 0
            self.plot_result()
        pass
