import numpy as np

import librosa
from .array import change_type


def load_audio(sample_rate: int):
    return lambda path: librosa.load(path, sr=sample_rate)[0]


def main(**params):
    parameter = params['parameter']
    retype = change_type(np.float32)

    def fun(audio):

        if params['audio_mode'] == "completely":
            y, sr = librosa.load(audio, sr=params['sample_rate'], res_type=params['resample'])
            duration = int(params['max_seconds'] * sr)

            if duration > len(y):
                if params['fill_mode'] == 'last_millisecond':
                    y = np.concatenate((
                        y, np.full((duration - y.shape[0]), y[-1])
                    ))
                elif params['fill_mode'] == 'loop':
                    while len(y) < duration:
                        y = np.concatenate((
                            y, y[:duration - len(y)]
                        ))
            else:
                y = y[:duration]

            if parameter in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                array = getattr(librosa.feature, parameter)(y=y, sr=sr)
            elif parameter == 'rms':
                array = getattr(librosa.feature, parameter)(y=y)[0]
            elif parameter == 'zero_crossing_rate':
                array = getattr(librosa.feature, parameter)(y=y)
            elif parameter == 'audio_signal':
                array = y

            array = np.array(array)
            if len(array.shape) == 2:
                array = array.transpose()
            array = retype(array)

            if params['scaler'] != 'no_scaler' and params.get('preprocess'):
                orig_shape = array.shape
                array = params['preprocess'].transform(array.reshape(-1, 1))
                array = array.reshape(orig_shape)

            array = array[np.newaxis, ...]

            return array

        elif params['audio_mode'] == "length_and_step":
            arrays = []
            au, sr = librosa.load(audio, sr=params['sample_rate'], res_type=params['resample'])
            cur_step = 0
            stop_flag = False
            duration = int(params['length'] * sr)

            while not stop_flag:

                y = au[cur_step:cur_step + int((params["length"]*1000))]
                if duration > len(y):
                    if params['fill_mode'] == 'last_millisecond':
                        y = np.concatenate((
                            y, np.full((duration - y.shape[0]), y[-1])
                        ))
                    elif params['fill_mode'] == 'loop':
                        while len(y) < duration:
                            y = np.concatenate((
                                y, y[:duration - len(y)]
                            ))

                if parameter in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
                    array = getattr(librosa.feature, parameter)(y=y, sr=sr)
                elif parameter == 'rms':
                    array = getattr(librosa.feature, parameter)(y=y)[0]
                elif parameter == 'zero_crossing_rate':
                    array = getattr(librosa.feature, parameter)(y=y)
                elif parameter == 'audio_signal':
                    array = y

                array = np.array(array)
                if len(array.shape) == 2:
                    array = array.transpose()
                array = retype(array)

                if params['scaler'] != 'no_scaler' and params.get('preprocess'):
                    orig_shape = array.shape
                    array = params['preprocess'].transform(array.reshape(-1, 1))
                    array = array.reshape(orig_shape)

                arrays.append(array)
                cur_step += int(params['step']*1000)
                if cur_step + (params['length']*1000) > len(au):
                    stop_flag = True

            arrays = np.array(arrays)
            arrays = retype(arrays)

            # print("arrays.shape", arrays.shape)
            return arrays
    return fun
