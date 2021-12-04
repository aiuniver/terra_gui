import psutil
import time
import pynvml as N
import numpy as np

from config import settings

from terra_ai import progress
from terra_ai.callbacks.utils import print_error, loss_metric_config, YOLO_ARCHITECTURE
from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.data.training.extra import CheckpointTypeChoice,StateStatusChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.training.training_history import History
from terra_ai.callbacks import interactive


class FitCallback:
    """CustomCallback for all task type"""

    def __init__(self, dataset: PrepareDataset, training_details: TrainingDetailsData,
                 model_name: str = "model", deploy_type: str = ""):
        super().__init__()
        print('\n FitCallback')
        self.name = "FitCallback"
        self.current_logs = {}
        self.usage_info = MemoryUsage(debug=False)
        self.training_detail = training_details
        self.training_detail.logs = None
        self.dataset = dataset
        self.dataset_path = dataset.data.path
        self.deploy_type = getattr(DeployTypeChoice, deploy_type)
        self.is_yolo = True if dataset.data.architecture in YOLO_ARCHITECTURE else False
        self.batch_size = training_details.base.batch
        self.nn_name = model_name
        self.deploy_path = training_details.deploy_path
        self.model_path = training_details.model_path
        self.stop_training = False

        self.history = History(dataset=dataset, training_details=training_details, deploy_type=self.deploy_type)

        self.batch = 0
        self.num_batches = 0
        self.retrain_epochs = self.training_detail.base.epochs
        self.last_epoch = self.history.last_epoch
        self.total_epochs = self.history.sum_epoch
        self.still_epochs = self.history.epochs
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self._sum_epoch_time = 0
        self.progress_name = "training"
        self.result = {
            'info': None,
            "train_usage": {
                "hard_usage": self.usage_info.get_usage(),
                "timings": {
                    "estimated_time": 0,
                    "elapsed_time": 0,
                    "still_time": 0,
                    "avg_epoch_time": 0,
                    "elapsed_epoch_time": 0,
                    "still_epoch_time": 0,
                    "epoch": {
                        "current": 0,
                        "total": 0
                    },
                    "batch": {
                        "current": 0,
                        "total": 0
                    },
                }
            },
            'train_data': None,
            'states': {}
        }
        # аттрибуты для чекпоинта
        self.checkpoint_config = training_details.base.architecture.parameters.checkpoint
        self.checkpoint_mode = self._get_checkpoint_mode()  # min max
        self.num_outputs = len(self.dataset.data.outputs.keys())
        self.metric_checkpoint = self.checkpoint_config.metric_name  # "val_mAP50" if self.is_yolo else "loss"

        self.samples_train = []
        self.samples_val = []
        self.samples_target_train = []
        self.samples_target_val = []

    def _get_checkpoint_mode(self):
        method_name = '_get_checkpoint_mode'
        try:
            print(method_name, self.checkpoint_config)
            if self.checkpoint_config.type == CheckpointTypeChoice.Loss:
                return 'min'
            elif self.checkpoint_config.type == CheckpointTypeChoice.Metrics:
                metric_name = self.checkpoint_config.metric_name
                return loss_metric_config.get("metric").get(metric_name).get("mode")
            else:
                print('\nClass FitCallback method _get_checkpoint_mode: No checkpoint types are found\n')
                return None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _logs_predict_extract(logs, prefix):
        pred_on_batch = []
        for key in logs.keys():
            if key.startswith(prefix):
                pred_on_batch.append(logs[key])
        return pred_on_batch

    def _best_epoch_monitoring(self):
        method_name = '_best_epoch_monitoring'
        try:
            print(method_name)
            output = "output" if self.is_yolo else f"{self.checkpoint_config.layer}"
            checkpoint_type = self.checkpoint_config.type.name.lower()
            metric = self.checkpoint_config.metric_name.name
            indicator = self.checkpoint_config.indicator.name.lower()
            checkpoint_list = self.history.get_checkpoint_data(output, checkpoint_type, metric, indicator)
            if self.checkpoint_mode == 'min' and checkpoint_list[-1] == min(checkpoint_list):
                return True
            elif self.checkpoint_mode == "max" and checkpoint_list[-1] == max(checkpoint_list):
                return True
            else:
                return False
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _set_result_data(self, param: dict) -> None:
        method_name = '_set_result_data'
        try:
            for key in param.keys():
                if key in self.result.keys():
                    self.result[key] = param[key]
                elif key == "timings":
                    self.result["train_usage"]["timings"]["estimated_time"] = param[key][0]
                    self.result["train_usage"]["timings"]["elapsed_time"] = param[key][1]
                    self.result["train_usage"]["timings"]["still_time"] = param[key][2]
                    self.result["train_usage"]["timings"]["avg_epoch_time"] = int(
                        self._sum_epoch_time / self.last_epoch)
                    self.result["train_usage"]["timings"]["elapsed_epoch_time"] = param[key][3]
                    self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][4]
                    self.result["train_usage"]["timings"]["epoch"] = param[key][5]
                    self.result["train_usage"]["timings"]["batch"] = param[key][6]
            self.result["train_usage"]["hard_usage"] = self.usage_info.get_usage()
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_result_data(self):
        return self.result

    def _get_train_status(self) -> str:
        return self.training_detail.state.status

    def _get_predict(self, current_model=None):
        method_name = '_get_predict'
        try:
            print(method_name)
            # current_model = deploy_model if deploy_model else self.model
            if self.is_yolo:
                current_predict = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_val)]
                current_target = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_target_val)]
            else:
                # TODO: настроить вывод массивов их обучения, выводить словарь
                #  {'train_true': train_true, 'train_pred': train_pred, 'val_true': val_true, 'val_pred': val_pred}
                if self.dataset.data.use_generator:
                    current_predict = current_model.predict(
                        self.dataset.dataset.get('val').batch(1), batch_size=1)
                else:
                    current_predict = current_model.predict(self.dataset.X.get('val'), batch_size=self.batch_size)
                # current_predict = None
                current_target = None
            return current_predict, current_target
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _estimate_step(current, start, now):
        method_name = '_estimate_step'
        try:
            # print(method_name)
            if current:
                _time_per_unit = (now - start) / current
            else:
                _time_per_unit = (now - start)
            return _time_per_unit
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def eta_format(eta):
        method_name = 'eta_format'
        try:
            print(method_name)
            if eta > 3600:
                eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                                         (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
            else:
                eta_format = '%d сек' % eta
            return ' %s' % eta_format
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def update_progress(self, target, current, start_time, finalize=False, stop_current=0, stop_flag=False):
        method_name = 'update_progress'
        try:
            # print(method_name)
            """
            Updates the progress bar.
            """
            _now_time = time.time()
            if finalize:
                eta = _now_time - start_time
            else:
                time_per_unit = self._estimate_step(current, start_time, _now_time)
                if stop_flag:
                    eta = time_per_unit * (target - current)
                else:
                    eta = time_per_unit * target
            return int(eta)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def is_best(self):
        return self._best_epoch_monitoring()

    def on_train_begin(self):
        method_name = 'on_train_begin'
        try:
            print(method_name, self.dataset.dataframe.keys())
            status = self._get_train_status()
            self._start_time = time.time()
            if status != StateStatusChoice.addtrain:
                self.batch = 1
            if not self.dataset.data.use_generator:
                if len(list(self.dataset.X['train'].values())[0]) % self.batch_size:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size + 1
                else:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size
            else:
                if len(self.dataset.dataframe['train']) % self.batch_size:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size + 1
                else:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_begin(self):
        print('on_epoch_begin')
        self.last_epoch += 1
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, arrays=None, train_data_idxs=None):
        method_name = 'on_train_batch_end'
        try:
            if self._get_train_status() == StateStatusChoice.stopped:
                print('_get_train_status() == "stopped"')
                self.stop_training = True
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.total_epochs if self._get_train_status() == StateStatusChoice.addtrain else self.retrain_epochs) * 100,
                    message="Обучение остановлено пользователем, ожидайте остановку...",
                    finished=False,
                )
            else:
                self.batch = batch
                msg_batch = {"current": batch, "total": self.num_batches}
                msg_epoch = {"current": self.last_epoch,
                             "total": self.total_epochs if self._get_train_status() == StateStatusChoice.addtrain
                             else self.retrain_epochs}
                still_epoch_time = self.update_progress(self.num_batches, self.batch,
                                                        self._time_first_step, stop_flag=True)
                elapsed_epoch_time = time.time() - self._time_first_step
                elapsed_time = time.time() - self._start_time
                target_ = self.num_batches * self.still_epochs
                still_current = self.num_batches * (self.total_epochs - self.last_epoch + 1)
                estimated_time = self.update_progress(target_, target_ - still_current + self.batch, self._start_time)

                still_time = self.update_progress(
                    target_, target_ - still_current + self.batch, self._start_time, stop_flag=True)

                if interactive.urgent_predict:
                    train_batch_data = interactive.update_state(arrays=arrays, train_idx=train_data_idxs)
                else:
                    train_batch_data = interactive.update_state(arrays=None, train_idx=None)
                if train_batch_data:
                    result_data = {
                        'timings': [estimated_time, elapsed_time, still_time,
                                    elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch],
                        'train_data': train_batch_data
                    }
                else:
                    result_data = {'timings': [estimated_time, elapsed_time, still_time,
                                               elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch]}

                self._set_result_data(result_data)
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.total_epochs if self._get_train_status() == StateStatusChoice.addtrain else self.retrain_epochs
                    ) * 100,
                    message=f"Обучение. Эпоха {self.last_epoch} из "
                            f"{self.total_epochs if self._get_train_status() in [StateStatusChoice.addtrain, StateStatusChoice.stopped] else self.retrain_epochs}",
                    finished=False,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_end(self, epoch, arrays=None, logs=None, train_data_idxs=None):
        method_name = 'on_epoch_end'
        try:
            # print(method_name, epoch)
            if self.is_yolo:
                self.history.current_logs = logs
            else:
                self.history.current_basic_logs(epoch=epoch, arrays=arrays, train_idx=train_data_idxs)
            self.history.update_log_history()
            print('\n', method_name, epoch, '\n', self.history.get_history())
            if epoch == 1:
                interactive.log_history = self.history.get_history()
            current_epoch_time = time.time() - self._time_first_step
            self._sum_epoch_time += current_epoch_time
            train_epoch_data = interactive.update_state(
                fit_logs=self.history.get_history(),
                arrays=arrays,
                current_epoch_time=current_epoch_time,
                on_epoch_end_flag=True,
                train_idx=train_data_idxs
            )
            self._set_result_data({'train_data': train_epoch_data})
            progress.pool(
                self.progress_name,
                percent=self.last_epoch / (
                    self.total_epochs
                    if self._get_train_status() == StateStatusChoice.addtrain or self._get_train_status() == StateStatusChoice.stopped else self.retrain_epochs
                ) * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из "
                        f"{self.total_epochs if self._get_train_status() in [StateStatusChoice.addtrain, StateStatusChoice.stopped] else self.retrain_epochs}",
                finished=False,
            )
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_train_end(self):
        method_name = 'on_train_end'
        try:
            print(method_name)
            if self.stop_training:
                interactive.addtrain_epochs.append(self.last_epoch-1)
            else:
                interactive.addtrain_epochs.append(self.last_epoch)
            if (self.stop_training and self.last_epoch < self.training_detail.base.epochs) or \
                    (self._get_train_status() == StateStatusChoice.trained and
                     self.last_epoch == self.training_detail.base.epochs):
                self.history.sum_epoch = self.training_detail.base.epochs
            self.history.save_logs()

            time_end = self.update_progress(
                self.num_batches * self.retrain_epochs, self.batch, self._start_time, finalize=True)
            self._sum_time += time_end
            total_epochs = self.total_epochs \
                if self._get_train_status() in [StateStatusChoice.addtrain,
                                                StateStatusChoice.stopped] else self.retrain_epochs
            if self.stop_training:
                progress.pool(
                    self.progress_name,
                    message=f"Обучение остановлено. Эпоха {self.last_epoch - 1} из {total_epochs}. Модель сохранена.",
                    data=self._get_result_data(),
                    finished=True,
                )
            else:
                percent = self.last_epoch / (
                    self.total_epochs if self._get_train_status() == StateStatusChoice.addtrain or
                                         self._get_train_status() == StateStatusChoice.stopped
                    else self.retrain_epochs
                ) * 100
                print('percent', percent, self.progress_name)

                self.training_detail.state.set("trained")
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=percent,
                    message=f"Обучение завершено. Эпоха {self.last_epoch} из {total_epochs}",
                    finished=True,
                )
                print(progress.pool(self.progress_name))
        except Exception as e:
            print_error('FitCallback', method_name, e)


# noinspection PyBroadException
class MemoryUsage:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            N.nvmlInit()
            self.gpu = settings.USE_GPU
        except:
            self.gpu = False

    def get_usage(self):
        usage_dict = {}
        if self.gpu:
            gpu_name = N.nvmlDeviceGetName(N.nvmlDeviceGetHandleByIndex(0))
            gpu_utilization = N.nvmlDeviceGetUtilizationRates(N.nvmlDeviceGetHandleByIndex(0))
            gpu_memory = N.nvmlDeviceGetMemoryInfo(N.nvmlDeviceGetHandleByIndex(0))
            usage_dict["GPU"] = {
                'gpu_name': gpu_name,
                'gpu_utilization': f'{gpu_utilization.gpu: .2f}',
                'gpu_memory_used': f'{gpu_memory.used / 1024 ** 3: .2f}GB',
                'gpu_memory_total': f'{gpu_memory.total / 1024 ** 3: .2f}GB'
            }
            if self.debug:
                print(f'GPU usage: {gpu_utilization.gpu: .2f} ({gpu_memory.used / 1024 ** 3: .2f}GB / '
                      f'{gpu_memory.total / 1024 ** 3: .2f}GB)')
        else:
            cpu_usage = psutil.cpu_percent(percpu=True)
            usage_dict["CPU"] = {
                'cpu_utilization': f'{sum(cpu_usage) / len(cpu_usage): .2f}',
            }
            if self.debug:
                print(f'Average CPU usage: {sum(cpu_usage) / len(cpu_usage): .2f}')
                print(f'Max CPU usage: {max(cpu_usage): .2f}')
        usage_dict["RAM"] = {
            'ram_utilization': f'{psutil.virtual_memory().percent: .2f}',
            'ram_memory_used': f'{psutil.virtual_memory().used / 1024 ** 3: .2f}GB',
            'ram_memory_total': f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB'
        }
        usage_dict["Disk"] = {
            'disk_utilization': f'{psutil.disk_usage("/").percent: .2f}',
            'disk_memory_used': f'{psutil.disk_usage("/").used / 1024 ** 3: .2f}GB',
            'disk_memory_total': f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB'
        }
        if self.debug:
            print(f'RAM usage: {psutil.virtual_memory().percent: .2f} '
                  f'({psutil.virtual_memory().used / 1024 ** 3: .2f}GB / '
                  f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB)')
            print(f'Disk usage: {psutil.disk_usage("/").percent: .2f} '
                  f'({psutil.disk_usage("/").used / 1024 ** 3: .2f}GB / '
                  f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB)')
        return usage_dict
