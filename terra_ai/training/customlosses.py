import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

__version__ = 0.08

from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.settings import CALLBACK_CLASSIFICATION_TREASHOLD_VALUE


class DiceCoef(tf.keras.metrics.Metric):

    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoef, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        # pass

    def update_state(self, y_true, y_pred, smooth=1, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        if len(shape) == 3:
            axis = [1, 2]
        elif len(shape) == 4:
            axis = [1, 2, 3]
        else:
            axis = [1]
        intersection = K.sum(y_true * y_pred, axis=axis)
        union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
        self.dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(DiceCoef, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.dice

    def reset_state(self):
        self.dice: float = 0
        # pass


class BalancedDiceCoef(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_dice_coef', encoding='ohe', **kwargs):
        super(BalancedDiceCoef, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.encoding = encoding
        # pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.encoding == 'ohe':
            y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        elif self.encoding == 'multi':
            y_pred = tf.where(y_pred > 0.9, 1., 0.)
        else:
            y_pred = tf.where(y_pred > 0.5, 1., 0.)

        balanced_dice = tf.convert_to_tensor(0., dtype=tf.float32)
        for i in range(y_true.shape[-1]):
            intersection = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
            union = K.sum(y_true[..., i:i + 1]) + K.sum(y_pred[..., i:i + 1])
            balanced_dice = tf.add(balanced_dice, tf.convert_to_tensor((2. * intersection + smooth) / (union + smooth)))
        self.dice = tf.convert_to_tensor(balanced_dice / y_true.shape[-1])
        # print(self.dice, intersection, union)

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedDiceCoef, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.dice

    def reset_state(self):
        self.dice: float = 0


# class RecallPercent(tf.keras.metrics.Metric):
#
#     def __init__(self, name='recall_percent', **kwargs):
#         super(RecallPercent, self).__init__(name=name, **kwargs)
#         self.recall: float = 0
#         # pass
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
#         recall = K.sum(y_true * y_pred)
#         total = K.sum(y_true)
#         self.recall = tf.convert_to_tensor(recall * 100 / total)
#
#     def get_config(self):
#         """
#         Returns the serializable config of the metric.
#         """
#         config = super(RecallPercent, self).get_config()
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def result(self):
#         return self.recall
#
#     def reset_state(self):
#         self.recall: float = 0
#         # pass


class BalancedRecall(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_recall', **kwargs):
        super(BalancedRecall, self).__init__(name=name, **kwargs)
        self.recall: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            recall = K.sum(y_true * y_pred)
            total = K.sum(y_true)
            self.recall = tf.convert_to_tensor(recall * 100 / total)
        else:
            balanced_recall = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                recall = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total = K.sum(y_true[..., i:i + 1])
                if total == tf.convert_to_tensor(0.):
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(0.))
                else:
                    balanced_recall = tf.add(balanced_recall, tf.convert_to_tensor(recall * 100 / total))
            self.recall = tf.convert_to_tensor(balanced_recall / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedRecall, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall: float = 0
        # pass


# class PrecisionPercent(tf.keras.metrics.Metric):
#
#     def __init__(self, name='precision_percent', **kwargs):
#         super(PrecisionPercent, self).__init__(name=name, **kwargs)
#         self.precision: float = 0
#         # pass
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.cast(y_pred, tf.float32)
#         y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
#         true_guess = K.sum(y_true * y_pred)
#         total = K.sum(y_pred)
#         self.precision = tf.convert_to_tensor(true_guess * 100 / total)
#
#     def get_config(self):
#         """
#         Returns the serializable config of the metric.
#         """
#         config = super(PrecisionPercent, self).get_config()
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)
#
#     def result(self):
#         return self.precision
#
#     def reset_state(self):
#         self.precision: float = 0
#         # pass


class BalancedPrecision(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_precision', **kwargs):
        super(BalancedPrecision, self).__init__(name=name, **kwargs)
        self.precision: float = 0
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            true_guess = K.sum(y_true * y_pred)
            # total_true = K.sum(y_true)
            total_pred = K.sum(y_pred)
            self.precision = tf.convert_to_tensor(true_guess * 100 / total_pred)
        else:
            balanced_precision = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                true_guess = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total = K.sum(y_pred[..., i:i + 1])
                if total == tf.convert_to_tensor(0.):
                    balanced_precision = tf.add(balanced_precision, tf.convert_to_tensor(0.))
                else:
                    balanced_precision = tf.add(balanced_precision, tf.convert_to_tensor(true_guess * 100 / total))
            self.precision = tf.convert_to_tensor(balanced_precision / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedPrecision, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.precision

    def reset_state(self):
        self.precision: float = 0
        # pass


class FScore(tf.keras.metrics.Metric):

    def __init__(self, name='f_score', **kwargs):
        super(FScore, self).__init__(name=name, **kwargs)
        self.score = tf.convert_to_tensor(0., dtype=tf.float32)
        # pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])
        true_guess = K.sum(y_true * y_pred)
        total_true = K.sum(y_true)
        total_pred = K.sum(y_pred)
        recall = tf.convert_to_tensor(true_guess / total_true, dtype=tf.float32)
        precision = tf.convert_to_tensor(true_guess / total_pred, dtype=tf.float32)
        self.score = tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall))
        # print('\nFScore', recall, precision, self.score)

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(FScore, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.score

    def reset_state(self):
        self.score: float = 0
        # pass


class BalancedFScore(tf.keras.metrics.Metric):

    def __init__(self, name='balanced_f_score', **kwargs):
        super(BalancedFScore, self).__init__(name=name, **kwargs)
        self.score = tf.convert_to_tensor(0., dtype=tf.float32)
        # pass

    def update_state(self, y_true, y_pred, show_class=False, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=y_true.shape[-1])

        if show_class:
            true_guess = K.sum(y_true * y_pred)
            total_true = K.sum(y_true)
            total_pred = K.sum(y_pred)
            recall = tf.convert_to_tensor(true_guess / total_true)
            precision = tf.convert_to_tensor(true_guess / total_pred)
            self.score = tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall), dtype=tf.float32)
        else:
            balanced_score = tf.convert_to_tensor(0., dtype=tf.float32)
            for i in range(y_true.shape[-1]):
                true_guess = K.sum(y_true[..., i:i + 1] * y_pred[..., i:i + 1])
                total_true = K.sum(y_true[..., i:i + 1])
                total_pred = K.sum(y_pred[..., i:i + 1])
                if total_true == tf.convert_to_tensor(0.) or total_pred == tf.convert_to_tensor(0.):
                    balanced_score = tf.add(balanced_score, tf.convert_to_tensor(0.))
                else:
                    recall = tf.convert_to_tensor(true_guess / total_true)
                    precision = tf.convert_to_tensor(true_guess / total_pred)
                    balanced_score = tf.add(
                        balanced_score, tf.convert_to_tensor(2 * precision * recall * 100 / (precision + recall))
                    )
            self.score = tf.convert_to_tensor(balanced_score / y_true.shape[-1], dtype=tf.float32)
            # self.score = tf.add(balanced_score, self.score)
            # self.score = tf.convert_to_tensor(balanced_score / y_true.shape[-1])

    def get_config(self):
        """
        Returns the serializable config of the metric.
        """
        config = super(BalancedFScore, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def result(self):
        return self.score

    def reset_state(self):
        self.score: float = 0
        # pass


class UnscaledMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='unscaled_mae', **kwargs):
        super(UnscaledMAE, self).__init__(name=name, **kwargs)

    @staticmethod
    def unscale_result(mae_result, output: int, dataset: CreatePreprocessing):
        preprocess_dict = dataset.preprocessing.get(output)
        target_key = None
        for i, column in enumerate(preprocess_dict.keys()):
            if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                target_key = column
            else:
                target_key = None
                break
        if target_key:
            result = np.expand_dims(mae_result, axis=-1)
            preset = {output: {target_key: result}}
            unscale = np.array(list(dataset.inverse_data(preset)[output].values()))
            try:
                return unscale.item()
            except ValueError:
                return unscale.squeeze().tolist()
        else:
            return mae_result


def YoloLoss(inputs, num_anchors,):
    """
    Функция подсчета ошибки.
        Входные параметры:
            inputs - Входные данные
            num_anchors - общее количество анкоров
            num_classes - количеств классов
    """
    # Массив используемых анкоров (в пикселях). Используетя по 3 анкора на каждый из 3 уровней сеток
    # данные значения коррелируются с размерностью входного изображения input_shape
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    # num_anchors = len(anchors)  # Сохраняем количество анкоров
    ignore_thresh = .5  # Порог вероятности обнаружения объекта
    num_layers = num_anchors // 3  # Подсчитываем количество анкоров на каждом уровне сетки
    y_pred = inputs[:num_layers]  # Из входных данных выцепляем посчитанные моделью значения
    y_true = inputs[num_layers:]  # Из входных данных выцепляем эталонные значения
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # Задаем маску анкоров для каждого уровня сеток

    def calc_iou(input1, input2):
        """
            Функция подсчета IoU
        """
        input1 = K.expand_dims(input1, -2)  # Добавляем одну размерность
        xy1 = input1[..., :2]  # Получаем координаты x,y центра
        wh1 = input1[..., 2:4]  # Получаем значения высоты и ширины
        wh_half1 = wh1 / 2.  # Делим значения высоты и ширины пополам
        top_left1 = xy1 - wh_half1  # Получаем значение, соответствующее верхнему левому углу
        right_bottom1 = xy1 + wh_half1  # Получаем значение, соотвествующее правому нижнему углу

        input2 = K.expand_dims(input2, 0)  # Добавляем одну размерность
        xy2 = input2[..., :2]  # Получаем координаты x,y центра
        wh2 = input2[..., 2:4]  # Получаем значения высоты и ширины
        wh_half2 = wh2 / 2.  # Делим значения высоты и ширины пополам
        top_left2 = xy2 - wh_half2  # Получаем значение, соответствующее верхнему левому углу
        right_bottom2 = xy2 + wh_half2  # Получаем значение, соотвествующее правому нижнему углу

        intersect_mins = K.maximum(top_left1, top_left2)  # Берем максимальные координаты из левых верхних углов
        intersect_maxes = K.minimum(right_bottom1,
                                    right_bottom2)  # Берем Минимальные координаты координаты из правых нижних углов
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # Считаем ширину и высоту области пересечения
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # Считаем площадь области пересечения

        area1 = wh1[..., 0] * wh1[..., 1]  # Считаем площадь первых элементов
        area2 = wh2[..., 0] * wh2[..., 1]  # Считаем площадь вторых элементов

        return intersect_area / (area1 + area2 - intersect_area)  # Возвращаем IoU

    # Получаем размерность входного изображения ( (13 х 13) * 32 = (416 х 416)) и приводим к типу элемента y_true[0]
    input_shape = K.cast(K.shape(y_pred[0])[1:3] * 32, K.dtype(y_true[0]))

    # Получаем двумерный массив, соответствующий размерностям сеток ((13, 13), (26, 26), (52, 52))
    grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    loss = 0  # Значение ошибки

    # Считываем количество элементов
    m = K.shape(y_pred[0])[0]  # Размер пакета
    batch_size = K.cast(m, K.dtype(y_pred[0]))  # Преобразуем к типу y_pred[0]

    for l in range(num_layers):  # Пробегаем по всем трем уровням сеток
        # Получаем маску для сетки l-го уровня по вероятности определения объекта (5-ый параметр в списке общих
        # параметров). В массиве object_mask будут значения, которые соответствуют только вероятности обнаружения
        # объекта
        object_mask = y_true[l][..., 4:5]  # Вернется набор данных вида: ([0][0][0][0]...[1]...[0])

        # Получаем аналогичную выборку для сетки l-го уровня с OHE (где записана позиция нашего класса) В массиве
        # true_class будут значения, которые соответсвуют только OHE представлению класса ядля данного уровня анкоров
        true_class = y_true[l][..., 5:]  # Вернется набор данных вида: ([0][0][0][0]...[1]...[0])

        num_sub_anchors = len(anchors[anchor_mask[l]])  # Получаем количество анкоров для отдельного уровян сетки (3)

        # Решейпим анкоры отдельного уровня сетки и записываем в переменную anchors_tensor
        anchors_tensor = K.reshape(K.constant(anchors[anchor_mask[l]]), [1, 1, 1, num_sub_anchors, 2])

        # Создаем двумерный массив grid со значениями [[[0, 0] , [0, 1] , [0, 2] , ... , [0, k]],
        #                                             [[1, 0] , [1, 1] , [1, 2] , ... , [1 ,k]],
        #                                             ...
        #                                             [[k, 0] , [k, 1] , [k, 2] , ... , [k, k]]]
        # где k - размерность сетки. Массив хранит индексы ячеек сетки
        grid_shape = K.shape(y_pred[l])[1:3]  # Получаем ширину и высоту сетки
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                        [1, grid_shape[1], 1, 1])  # Создаем вертикальную линию
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                        [grid_shape[0], 1, 1, 1])  # Создаем горизонтальную линию
        grid = K.concatenate([grid_x, grid_y])  # Объединяем
        grid = K.cast(grid, K.dtype(y_pred[l]))  # Приводим к типу y_pred[l]

        # Решейпим y_pred[l]
        # feats = K.reshape(y_pred[l], [-1, grid_shape[0], grid_shape[1], num_sub_anchors, num_classes + 5])
        feats = y_pred[l]

        # Считаем ошибку в определении координат центра объекта
        # Получаем координаты центра объекта из спредиктенного значения
        pred_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
        # Производим обратные вычесления для оригинальных значений из y_true для координат центра объекта
        true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid  # Реальные координаты центра bounding_box
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]  # чем больше бокс, тем меньше ошибка
        # binary_crossentropy для истинного значения и спредиктенного (object_mask для подсчета только требуемого
        # значения)
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(true_xy, feats[..., 0:2], from_logits=True)

        # Считаем ошибку в определении координат ширины и высоты
        # Получаем значения ширины и высоты изображения из спредиктенного значения
        pred_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
        # Производим обратные вычесления для оригинальных значений из y_true для ширины и высоты объекта
        true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        # Оставляем значение высоты и ширины только у тех элементов, где object_mask = 1
        true_wh = K.switch(object_mask, true_wh, K.zeros_like(true_wh))
        # Считаем значение ошибки в определении высоты и ширины
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh - feats[..., 2:4])

        # Объединяем значения в один  массив
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Считаем ошибку в определении обнаружения какого-либо класса Для этого вначале надо отсечь все найденные
        # объекты, вероятность которых меньше установленного значения ignore_thresh

        # Определяем массив, который будет хранить данные о неподходящих значениях
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')  # Приводим тип object_mask к типу 'bool'

        # Функция, определяющая данные, которые требуется игнорировать
        # Пробегаем по всем элементам пакета (b<m)
        # Получаем параметры реального bounding_box для текущей ячейки
        # Считаем IoU реального и спредиктенного
        # В зависимости от best_iou < ignore_thresh помечаем его как верно распознанный или неверено
        def loop_body(
                b,
                ignore_mask,
        ):
            # в true_box запишутся первые 4 параметра (центр, высота и ширина объекта) того элемента,
            # значение которого в object_mask_bool равно True
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # Подсчитываем iou для спредиктенной ограничивающей рамки (pred_box) и оригинальной (true_box)
            iou = calc_iou(pred_box[b], true_box)
            # Находим лучшую ограничивающую рамку
            best_iou = K.max(iou, axis=-1)
            # Записываем в ignore_mask true или false в зависимости от (best_iou < ignore_thresh)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask  # Увеличиваем счетчик на еденицу и возвращаем ignore_mask

        # Пробегаем в цикле по всем элементам в пределах значения m (m = batch size)
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()  # Приводим ignore_mask к тензору
        ignore_mask = K.expand_dims(ignore_mask, -1)  # Добавляем еще одну размерность в конце ignore_mask

        # Считаем значение ошибки
        # 1 компонента - для значений, которые были верно спредиктены
        # 2 компонента - для значения, которые были неверно спредиктены
        confidence_loss = (
                object_mask * K.binary_crossentropy(object_mask, feats[..., 4:5], from_logits=True) +
                (1 - object_mask) * K.binary_crossentropy(object_mask, feats[..., 4:5], from_logits=True) * ignore_mask
        )

        # Считаем ошибку в определении класса объекта
        class_loss = object_mask * K.binary_crossentropy(true_class, feats[..., 5:], from_logits=True)

        # Считаем суммарную ошибку
        xy_loss = K.sum(xy_loss) / batch_size
        wh_loss = K.sum(wh_loss) / batch_size
        confidence_loss = K.sum(confidence_loss) / batch_size
        class_loss = K.sum(class_loss) / batch_size
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss  # Возвращаем значение ошибки
