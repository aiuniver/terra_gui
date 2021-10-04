import numpy as np

import librosa


def main(**params):
    parameter = params['parameter']

    def fun(audio):

        y, sr = librosa.load(audio, sr=params['sample_rate'], res_type=params['resample'])
        if params['sample_rate'] > len(y):
            zeros = np.zeros((params['sample_rate'] - len(y),))
            y = np.concatenate((y, zeros))

        if parameter in ['chroma_stft', 'mfcc', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']:
            array = getattr(librosa.feature, parameter)(y=y, sr=sr)
        elif parameter == 'rms':
            array = getattr(librosa.feature, parameter)(y=y)[0]
        elif parameter == 'zero_crossing_rate':
            array = getattr(librosa.feature, parameter)(y=y)
        elif parameter == 'audio_signal':
            array = y

        array = np.array(array)
        if array.dtype == 'float64':
            array = array.astype('float32')

        if params['scaler'] != 'no_scaler' and params.get('preprocess'):
            orig_shape = array.shape
            array = params['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        array = array[np.newaxis, ...]

        return array

    return fun
