import os
from collections import Counter

import librosa
import tensorflow_hub
import torch
import grpc
import wave
import copy
import json
import base64
import hmac
import numpy as np

from time import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from mutagen import mp3
from gtts import gTTS
from scipy.spatial.distance import cosine
from PIL import Image

from .common import _associate_detections_to_trackers, _non_max_suppression
from .advansed_services import _SpeechToTextStub, _KalmanBoxTracker
from .function_blocks import ChangeSize
from .input_blocks import Input, BaseInput
from .internal_out_blocks import ModelOutput
from .main_blocks import CascadeBlock, BaseBlock
from .advansed_services import _Extractor, _NearestNeighborDistanceMetric, _Tracker, _Detection
from .model_blocks import BaseModel
from .utils import stt_pb2


class BaseService(BaseBlock):

    def __init__(self, **kwargs):
        super().__init__()

        self.path: str = ""
        self.save_path: str = ""
        self.outs = {}

    def get_outputs(self):
        return list(self.outs.keys())

    def set_path(self, model_path: str, save_path: str):
        self.path = os.path.join(model_path, self.path, 'model')
        self.save_path = save_path

    def execute(self):
        pass


class Wav2Vec(BaseService):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        source = list(self.inputs.values())[0].execute()
        speech_array = librosa.load(source, sr=16000)[0]

        inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = self.processor.batch_decode(predicted_ids)

        return predicted_sentences[0].lower()


class TinkoffAPI(BaseService):

    def __init__(self, **kwargs):
        super().__init__()

        self.api_key: str = kwargs.get("api_key")
        self.secret_key: str = kwargs.get("secret_key")
        self.max_alternatives: int = kwargs.get("max_alternatives", 3)
        self.do_not_perform_vad: bool = kwargs.get("do_not_perform_vad", True)
        self.profanity_filter: bool = kwargs.get("profanity_filter", True)
        self.enable_automatic_punctuation: bool = True
        self.expiration_time: int = kwargs.get("expiration_time", int(6e4))
        self.endpoint: str = kwargs.get("endpoint", 'stt.tinkoff.ru:443')
        self.stub = _SpeechToTextStub(grpc.secure_channel(self.endpoint, grpc.ssl_channel_credentials()))
        self.metadata = self._authorization_metadata(self.api_key, self.secret_key,
                                                     "tinkoff.cloud.stt", self.expiration_time)

    def execute(self):
        source = list(self.inputs.values())[0].execute()
        __request = self._build_request(path=source, max_alternatives=self.max_alternatives,
                                        do_not_perform_vad=self.do_not_perform_vad,
                                        profanity_filter=self.profanity_filter,
                                        enable_automatic_punctuation=self.enable_automatic_punctuation)
        response = self.stub.Recognize(__request, metadata=self.metadata)

        tinkoff_res = ''
        ch = '-  '
        for result in response.results:
            if int(result.channel) == int(0):
                ch = '-  '
            elif int(result.channel) == int(1):
                ch = '-- '

            for alternative in result.alternatives:
                tinkoff_res += '\n' + ch + alternative.transcript

        return tinkoff_res

    @staticmethod
    def _build_request(path: str, max_alternatives: int, do_not_perform_vad: bool, profanity_filter: bool,
                       enable_automatic_punctuation: bool):
        request = stt_pb2.RecognizeRequest()
        if path.split('.')[-1].lower() == 'mp3':
            mp3_file = mp3.MP3(path)
            num_ch = int(mp3_file.info.channels)
            sr_audio = int(mp3_file.info.sample_rate)
            with open(path, "rb") as f:
                request.audio.content = f.read()
            request.config.encoding = stt_pb2.AudioEncoding.MPEG_AUDIO

        elif path.split('.')[-1].lower() == 'wav':
            with wave.open(path) as f:
                sr_audio = f.getframerate()
                num_ch = f.getnchannels()
                request.audio.content = f.readframes(f.getnframes())
            request.config.encoding = stt_pb2.AudioEncoding.LINEAR16

        request.config.sample_rate_hertz = sr_audio
        request.config.num_channels = num_ch  # количество каналов в записи

        request.config.max_alternatives = max_alternatives  # включение альтернативных распознаваний
        request.config.do_not_perform_vad = do_not_perform_vad  # отключение режима диалога
        request.config.profanity_filter = profanity_filter  # фильтр ненормативной лексики
        request.config.enable_automatic_punctuation = enable_automatic_punctuation  # вставка знаков пунктуации
        return request

    def _authorization_metadata(self, api_key: str, secret_key, scope, expiration_time=6000):
        auth_payload = {
            "iss": "test_issuer",
            "sub": "test_user",
            "aud": scope
        }

        metadata = [
            ("authorization", "Bearer " + self._generate_jwt(api_key, secret_key, auth_payload, expiration_time))
        ]
        return list(metadata)

    @staticmethod
    def _generate_jwt(api_key, secret_key, payload, expiration_time=6000):
        header = {
            "alg": "HS256",
            "typ": "JWT",
            "kid": api_key
        }
        payload_copy = copy.deepcopy(payload)
        current_timestamp = int(time())
        payload_copy["exp"] = current_timestamp + expiration_time

        payload_bytes = json.dumps(payload_copy, separators=(',', ':')).encode("utf-8")
        header_bytes = json.dumps(header, separators=(',', ':')).encode("utf-8")

        data = (base64.urlsafe_b64encode(header_bytes).strip(b'=') + b"." +
                base64.urlsafe_b64encode(payload_bytes).strip(b'='))

        signature = hmac.new(base64.urlsafe_b64decode(secret_key), msg=data, digestmod="sha256")
        jwt = data + b"." + base64.urlsafe_b64encode(signature.digest()).strip(b'=')
        return jwt.decode("utf-8")


class GoogleTTS(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.outs = {out.front_name: out for out in ModelOutput().get(type_='Text2Speech')}
        self.language = kwargs.get("language", "ru")

    def execute(self):
        if max([issubclass(type(input_), BaseModel) for input_ in list(self.inputs.values())]):
            outs_ = list(self.inputs.values())[0].get_outputs()
            idx = outs_.index('Текст')
            source = list(self.inputs.values())[0].execute()[idx]
        else:
            source = list(self.inputs.values())[0].execute()

        result = gTTS(source, lang=self.language)

        data = {
            'source': source,
            'model_predict': result,
            'options': {},
            'save_path': self.save_path
        }

        return [out().execute(**data) for name, out in self.outs.items()]


class YoloV5(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.version: str = f'yolov5{kwargs.get("version", "small")[0].lower()}'
        self.render_img: bool = kwargs.get("render_img", True)
        self.pretrained: bool = kwargs.get("pretrained", True)
        self.force_reload: bool = kwargs.get("force_reload", True)
        self.model_url = kwargs.get("model_url", 'ultralytics/yolov5')

        self.model = None

    def execute(self):
        if not self.model:
            self.model = torch.hub.load(self.model_url, self.version,
                                        pretrained=self.pretrained, force_reload=self.force_reload)

        frame = list(self.inputs.values())[0].execute()
        out_source = ''
        if isinstance(frame, str):
            out_source = frame
            frame = Image.open(frame)
        frame = np.squeeze(frame)
        out = self.model(frame)
        if self.render_img:
            return {'image_array': out.render()[0], 'source': out_source}

        return {'bboxes': out.xyxy[0].cpu().numpy(), 'source': out_source}


class DeepSort(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.min_confidence = kwargs.get("min_confidence", 0.3)
        self.nms_max_overlap = kwargs.get("nms_max_overlap", 1.0)
        self.height, self.width = None, None
        self.path = ''
        self.extractor = None

        max_cosine_distance = kwargs.get("max_dist", 0.2)
        max_iou_distance = kwargs.get("max_iou_distance", 0.7)
        nn_budget = kwargs.get("nn_budget", 100)
        n_init = kwargs.get("n_init", 3)
        deep_max_age = kwargs.get("deep_max_age", 70)
        metric = _NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = _Tracker(metric, max_iou_distance=max_iou_distance, max_age=deep_max_age, n_init=n_init)

    def set_path(self, model_path: str):
        self.path = model_path
        print(self.path)
        self.extractor = _Extractor(self.path)
        print(self.extractor)

    def execute(self):
        bbox_xyxy, ori_img = None, None

        for input_type in self.inputs.keys():
            if input_type in Input.__dict__.keys():
                ori_img = self.inputs.get(input_type).execute()
            else:
                bbox_xyxy = self.inputs.get(input_type).execute()
                if not len(bbox_xyxy):
                    return np.zeros((1, 5))

        confidences = bbox_xyxy[:, 4]
        bbox_xyxy = bbox_xyxy[:, :4].astype(int)
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xyxy, ori_img)

        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        detections = [_Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = _non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        else:
            outputs = np.zeros((1, 5))
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_tlwh = None
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_tlwh(det):
        det = det.copy()
        w = det[:, 2] - det[:, 0]
        h = det[:, 3] - det[:, 1]
        det[:, 2] = w
        det[:, 3] = h
        return det

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            try:
                features = self.extractor(im_crops)
            except Exception as e:
                print(e)
                raise e
        else:
            features = np.array([])
        return features


class Sort(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.max_age = kwargs.get("max_age", 5)
        self.min_hits = kwargs.get("min_hits", 3)
        self.trackers = []
        self.frame_count = 0

    def execute(self):
        """
            Params:
              dots - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            Requires: this method must be called once for each frame even with empty detections
                (use np.empty((0, 5)) for frames without detections).
            Returns the a similar array, where the last column is the object ID.
            NOTE: The number of objects returned may differ from the number of detections provided.
        """

        dets = list(self.inputs.values())[0].execute()
        if not len(dets):
            return np.empty((0, 5))

        self.frame_count += 1

        # get predicted locations from existing trackers.
        tracks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(tracks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = _associate_detections_to_trackers(dets, tracks)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = _KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class BiTBasedTracker(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.distance_threshold = kwargs.get("distance_threshold", 0.15)
        self.max_age = kwargs.get("max_age", 5)
        self.metric = kwargs.get("metric", "")
        self.peoples = {}
        self.people_count = 0
        self.module = tensorflow_hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1")

        self.resizer = ChangeSize((128, 128, 3))
        self.dead_tracks = Counter()

        self.start_flag = True

    def __call__(self, dets: np.ndarray, image: np.ndarray) -> np.ndarray:
        out = []
        crops = []
        dets = dets[:, :4].astype(np.int)

        for b in dets:
            if b[0] == b[2] or b[1] == b[3]:
                b[:2] -= 5
                b[2:] += 5
                b[b < 0] = 0

            i = image[b[1]: b[3], b[0]: b[2]]
            crops.append(self.resizer.execute(i))
        crops = np.array(crops).astype(np.float32)
        vectors = self.module(crops).numpy()

        ids = list(self.peoples.keys())

        if self.start_flag:
            self.peoples = {n: i for n, i in enumerate(vectors)}
            self.people_count = len(vectors)
            self.start_flag = not self.start_flag
        else:
            for new_vec, box in zip(vectors, dets):
                min_dist = 1
                min_id = 0
                for id in ids:
                    dist = cosine(self.peoples[id], new_vec)
                    if dist < min_dist:
                        min_dist = dist
                        min_id = id
                if min_dist < self.distance_threshold:
                    self.peoples[min_id] = np.mean(np.concatenate(
                        (new_vec[np.newaxis, ...], self.peoples[min_id][np.newaxis, ...]), axis=0
                    ), axis=0)
                    current_id = min_id
                    ids.remove(id)
                else:
                    self.peoples[self.people_count] = new_vec
                    current_id = self.people_count
                    self.people_count += 1

                out.append(list(box) + [current_id])

        for id in ids:
            self.dead_tracks[id] += 1
            if self.dead_tracks[id] >= self.max_age:
                del self.dead_tracks[id]
                del self.peoples[id]

        return np.array(out)


class Service(CascadeBlock):
    Wav2Vec = Wav2Vec
    TinkoffAPI = TinkoffAPI
    GoogleTTS = GoogleTTS
    YoloV5 = YoloV5
    DeepSort = DeepSort
    Sort = Sort
    BiTBasedTracker = BiTBasedTracker
