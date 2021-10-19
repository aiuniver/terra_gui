import numpy as np

import librosa
from .array import change_type


def main(**params):
    parameter = params['parameter']
    retype = change_type(np.float32)

    def fun(audio):

        y, sr = librosa.load(audio, sr=params['sample_rate'], res_type=params['resample'])

        duration = params['max_seconds'] * sr

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
        array = retype(array)

        if params['scaler'] != 'no_scaler' and params.get('preprocess'):
            orig_shape = array.shape
            array = params['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        array = array[np.newaxis, ...]

        return array

    return fun
