import os
import cv2
import numpy as np

from typing import Any

from terra_ai.datasets.utils import resize_frame
from terra_ai.data.datasets.extra import LayerScalerVideoChoice, LayerVideoFillModeChoice, LayerVideoModeChoice
from .base import Array


class VideoArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        video: list = []
        cur_step = 0
        instructions_paths = []
        out_array = []
        cascade_mode = True if dataset_folder is None else False

        for elem in sources:
            cap = cv2.VideoCapture(elem)
            if cap.isOpened():
                if options['video_mode'] == LayerVideoModeChoice.completely:
                    if cascade_mode:
                        video.append(elem)
                    else:
                        video.append(';'.join([elem, f'[{cur_step}-{options["max_frames"]}]']))
                elif options['video_mode'] == LayerVideoModeChoice.length_and_step:
                    cur_step = 0
                    stop_flag = False
                    cap = cv2.VideoCapture(elem)
                    frame_count = int(cap.get(7))
                    while not stop_flag:
                        video.append(';'.join([elem, f'[{cur_step}-{cur_step + options["length"]}]']))
                        cur_step += options['step']
                        if cur_step + options['length'] > frame_count:
                            stop_flag = True
                            if options['length'] < frame_count:
                                video.append(
                                    ';'.join([elem, f'[{frame_count - options["length"]}-{frame_count}]']))

        for elem in video:
            tmp_array = []
            output_movie = None
            if cascade_mode:
                path = elem
                slicing = [cur_step]
            else:
                path, slicing = elem.split(';')
                slicing = [int(x) for x in slicing[1:-1].split('-')]
            name, ext = os.path.splitext(os.path.basename(path))
            cap = cv2.VideoCapture(path)
            cap.set(1, slicing[0])
            orig_shape = (int(cap.get(3)), int(cap.get(4)))
            frames_number = 0
            if not cascade_mode:
                os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))),
                            exist_ok=True)
                save_path = os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                         f'{name}_[{slicing[0]}-{slicing[1]}]{ext}')
                instructions_paths.append(save_path)
                output_movie = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), int(cap.get(5)), orig_shape)
            stop_flag = False
            while not stop_flag:
                ret, frame = cap.read()
                if not ret or (not cascade_mode and frames_number == slicing[1] - slicing[0]):  # frames_number > frames_count or
                    stop_flag = True
                else:
                    if not cascade_mode:
                        output_movie.write(frame)
                    tmp_array.append(frame)
                    frames_number += 1
            if cascade_mode:
                options['max_frames'] = frames_number
            if options['video_mode'] == 'completely' and options['max_frames'] > frames_number or \
                    options['video_mode'] == 'length_and_step' and options['length'] > frames_number:
                fr_to_add, tot_frames = 0, 0
                if options['video_mode'] == 'completely':
                    fr_to_add = options['max_frames'] - frames_number
                    tot_frames = options['max_frames']
                elif options['video_mode'] == 'length_and_step':
                    fr_to_add = options['length'] - frames_number
                    tot_frames = options['length']
                frames_to_add = self.add_frames(video_array=np.array(tmp_array),
                                                fill_mode=options['fill_mode'],
                                                frames_to_add=fr_to_add,
                                                total_frames=tot_frames)
                for arr in frames_to_add:
                    if not cascade_mode:
                        output_movie.write(arr)
                    else:
                        tmp_array.append(arr)
            if not cascade_mode:
                output_movie.release()
            else:
                new_array = []
                for frame in tmp_array:
                    new_array.append(frame[:, :, [2, 1, 0]])
                out_array.append(np.array(new_array))

        instructions = {'instructions': instructions_paths if instructions_paths else out_array[0],
                        'parameters': {'height': options['height'],
                                       'width': options['width'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'min_scaler': options['min_scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'scaler': options['scaler'],
                                       'frame_mode': options['frame_mode'],
                                       'fill_mode': options['fill_mode'],
                                       'video_mode': options['video_mode'],
                                       'length': options['length'],
                                       'max_frames': options['max_frames']}
                        }

        return instructions

    def create(self, source: Any, **options):
        if isinstance(source, str):
            array = []
            slicing = [int(x) for x in source[source.index('[') + 1:source.index(']')].split('-')]
            frames_count = slicing[1] - slicing[0]
            cap = cv2.VideoCapture(source)
            try:
                for _ in range(frames_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = frame[:, :, [2, 1, 0]]
                    array.append(frame)
            finally:
                cap.release()

            array = np.array(array)
        else:
            array = source

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        trgt_shape = (options['height'], options['width'])
        resized_array = []
        for i in range(len(array)):
            if array[i].shape[1:-1] != trgt_shape:
                resized_array.append(resize_frame(image_array=array[i],
                                                  target_shape=trgt_shape,
                                                  frame_mode=options['frame_mode']))
        array = np.array(resized_array)

        if options['scaler'] != LayerScalerVideoChoice.no_scaler and options.get('preprocess'):
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array

    @staticmethod
    def add_frames(video_array, fill_mode, frames_to_add, total_frames):

        frames: np.ndarray = np.array([])

        if fill_mode == LayerVideoFillModeChoice.last_frames:
            frames = np.full((frames_to_add, *video_array[-1].shape), video_array[-1], dtype='uint8')
        elif fill_mode == LayerVideoFillModeChoice.average_value:
            mean = np.mean(video_array, axis=0, dtype='uint16')
            frames = np.full((frames_to_add, *mean.shape), mean, dtype='uint8')
        elif fill_mode == LayerVideoFillModeChoice.loop:
            current_frames = (total_frames - frames_to_add)
            if current_frames > frames_to_add:
                frames = np.flip(video_array[-frames_to_add:], axis=0)
            elif current_frames <= frames_to_add:
                for i in range(frames_to_add // current_frames):
                    frames = np.flip(video_array[-current_frames:], axis=0)
                    video_array = np.concatenate((video_array, frames), axis=0)
                if frames_to_add + current_frames != video_array.shape[0]:
                    frames = np.flip(video_array[-(frames_to_add + current_frames - video_array.shape[0]):], axis=0)
                    video_array = np.concatenate((video_array, frames), axis=0)
                frames = video_array[current_frames:]

        return frames
