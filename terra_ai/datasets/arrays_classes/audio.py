import os
import librosa
import numpy as np
import librosa.feature as librosa_feature

from typing import Any
from pydub import AudioSegment
from librosa import load as librosa_load

from terra_ai.data.datasets.extra import LayerAudioModeChoice, LayerScalerAudioChoice
from .base import Array


class AudioArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        audio_files: list = []
        instructions_paths = []
        audio_arrays = []

        for elem in sources:
            try:
                librosa_load(elem, duration=0.002, res_type='scipy')  # Проверка файла на аудио-формат.
                if options['audio_mode'] == LayerAudioModeChoice.completely:
                    audio_files.append(';'.join([elem, f'[0.0-{options["max_seconds"]}]']))
                elif options['audio_mode'] == LayerAudioModeChoice.length_and_step:
                    cur_step = 0.0
                    stop_flag = False
                    sample_length = AudioSegment.from_file(elem).duration_seconds
                    while not stop_flag:
                        audio_files.append(';'.join([elem, f'[{cur_step}-{round(cur_step + options["length"], 1)}]']))
                        cur_step += options['step']
                        cur_step = round(cur_step, 1)
                        if cur_step + options['length'] > sample_length:
                            stop_flag = True
            except:
                pass

        duration_ = options['max_seconds'] if options['audio_mode'] == 'completely' else options['length']

        for elem in audio_files:
            path, slicing = elem.split(';')
            name, ext = os.path.splitext(os.path.basename(path))
            slicing = [float(x) for x in slicing[1:-1].split('-')]
            duration = round(slicing[1] - slicing[0], 1)
            audio = AudioSegment.from_file(path, start_second=slicing[0], duration=duration)

            if round(duration - audio.duration_seconds, 3) != 0:
                while not round(duration - audio.duration_seconds, 3):
                    if options['fill_mode'] == 'last_millisecond':
                        audio = audio.append(audio[-2], crossfade=0)
                    elif options['fill_mode'] == 'loop':
                        duration_to_add = round(duration - audio.duration_seconds, 3)
                        if audio.duration_seconds < duration_to_add:
                            audio = audio.append(audio[0:audio.duration_seconds * 1000], crossfade=0)
                        else:
                            audio = audio.append(audio[0:duration_to_add * 1000], crossfade=0)

            if dataset_folder is not None:
                os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(path))),
                            exist_ok=True)
                save_path = os.path.join(dataset_folder, os.path.basename(os.path.dirname(path)),
                                         f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
                audio.export(save_path, format=ext[1:])
                instructions_paths.append(save_path)
            else:
                audio_arrays.append(audio)

        instructions = {'instructions': instructions_paths if instructions_paths else audio_arrays,
                        'parameters': {'duration': duration_,
                                       'audio_mode': options['audio_mode'],
                                       'max_length': options['max_length'],
                                       'length': options['length'],
                                       'step': options['step'],
                                       'sample_rate': options['sample_rate'],
                                       'resample': options['resample'],
                                       'parameter': options['parameter'],
                                       'cols_names': options['cols_names'],
                                       'scaler': options['scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'min_scaler': options['min_scaler'],
                                       'put': options['put']}}

        return instructions

    def create(self, source: Any, **options):

        array = []
        parameter = options['parameter']
        sample_rate = options['sample_rate']

        if isinstance(source, str):
            y, sr = librosa_load(path=source, sr=options.get('sample_rate'), res_type=options.get('resample'))
        else:
            orig_sr = source.frame_rate
            y = np.array(source.get_array_of_samples()).astype('float')  # ВНИМАТЕЛЬНО С FLOAT!!!!!!!!!@@@@@@@@@@@@@@
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sample_rate, res_type=options.get('resample'))

        if round(sample_rate * options['duration'], 0) > y.shape[0]:
            zeros = np.zeros((int(round(sample_rate * options['duration'], 0)) - y.shape[0],))
            y = np.concatenate((y, zeros))
        if parameter in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
            array = getattr(librosa_feature, parameter)(y=y, sr=sample_rate)
        elif parameter == 'rms':
            array = getattr(librosa_feature, parameter)(y=y)[0]
        elif parameter == 'zero_crossing_rate':
            array = getattr(librosa_feature, parameter)(y=y)
        elif parameter == 'audio_signal':
            array = y

        array = np.array(array)
        if len(array.shape) == 2:
            array = array.transpose()
        if array.dtype == 'float64':
            array = array.astype('float32')

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        if options['scaler'] != LayerScalerAudioChoice.no_scaler and options.get('preprocess'):
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)
        return array
