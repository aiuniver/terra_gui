import os

import librosa
import torch
import grpc
import wave
import copy
import json
import base64
import hmac
import numpy as np

from typing import Any
from time import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from mutagen import mp3

from terra_ai.cascades.input_blocks import Input
from terra_ai.cascades.main_blocks import CascadeBlock, BaseBlock
from terra_ai.cascades.service.tracking import _Extractor, _NearestNeighborDistanceMetric, _Tracker, _Detection, \
    _non_max_suppression
from terra_ai.cascades.utils import stt_pb2, longrunning_pb2


class BaseService(BaseBlock):

    def execute(self):
        pass


class _SpeechToTextStub(object):
    """Speech recognition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Recognize = channel.unary_unary(
            '/tinkoff.cloud.stt.v1.SpeechToText/Recognize',
            request_serializer=stt_pb2.RecognizeRequest.SerializeToString,
            response_deserializer=stt_pb2.RecognizeResponse.FromString,
        )
        self.StreamingRecognize = channel.stream_stream(
            '/tinkoff.cloud.stt.v1.SpeechToText/StreamingRecognize',
            request_serializer=stt_pb2.StreamingRecognizeRequest.SerializeToString,
            response_deserializer=stt_pb2.StreamingRecognizeResponse.FromString,
        )
        self.LongRunningRecognize = channel.unary_unary(
            '/tinkoff.cloud.stt.v1.SpeechToText/LongRunningRecognize',
            request_serializer=stt_pb2.LongRunningRecognizeRequest.SerializeToString,
            response_deserializer=longrunning_pb2.Operation.FromString,
        )


class Wav2Vec(BaseService):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def __init__(self):
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


class TextToSpeech(BaseService):

    def __init__(self):
        super().__init__()

    def execute(self):
        sources = []
        if isinstance(sources, str):
            sources = [sources]

        for source in sources:
            yield source


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
        frame = np.squeeze(frame)
        out = self.model(frame)
        if self.render_img:
            return out.render()[0]

        return out.xyxy[0].cpu().numpy()


class DeepSort(BaseService):

    def __init__(self, **kwargs):
        super().__init__()
        self.min_confidence = kwargs.get("min_confidence", 0.3)
        self.nms_max_overlap = kwargs.get("nms_max_overlap", 1.0)
        self.height, self.width = None, None
        model_path = kwargs.get("model_path", "")
        if not os.path.isabs(model_path):
            parent_dir = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
            model_path = os.path.join(parent_dir, model_path)
        self.extractor = _Extractor(model_path)

        max_cosine_distance = kwargs.get("max_dist", 0.2)
        max_iou_distance = kwargs.get("max_iou_distance", 0.7)
        nn_budget = kwargs.get("nn_budget", 100)
        n_init = kwargs.get("n_init", 3)
        deep_max_age = kwargs.get("deep_max_age", 70)
        metric = _NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = _Tracker(metric, max_iou_distance=max_iou_distance, max_age=deep_max_age, n_init=n_init)

    def execute(self):
        bbox_xyxy, ori_img = None, None
        print(self.inputs)
        for input_type in self.inputs.keys():
            if input_type in Input.__dict__.keys():
                ori_img = self.inputs.get(input_type).execute()
            else:
                bbox_xyxy = self.inputs.get(input_type).execute()
                print("DEEP: ", type(bbox_xyxy), bbox_xyxy.shape, bbox_xyxy)
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
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


class Service(CascadeBlock):
    Wav2Vec = Wav2Vec
    TinkoffAPI = TinkoffAPI
    TextToSpeech = TextToSpeech
    YoloV5 = YoloV5
    DeepSort = DeepSort
