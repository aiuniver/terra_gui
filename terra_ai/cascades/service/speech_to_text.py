from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from ..utils import stt_pb2
import grpc
import copy
from time import time
import json
import base64
from mutagen import mp3
import hmac


def wav2vec2_large_russian(**nothing):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")

    def fun(speech_array):
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)

        return predicted_sentences[0].lower()

    return fun


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
                response_deserializer=stt_pb2.Operation.FromString,
                )


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


def _authorization_metadata(api_key: str, secret_key, scope, expiration_time):
    auth_payload = {
        "iss": "test_issuer",
        "sub": "test_user",
        "aud": scope
    }

    metadata = [
        ("authorization", "Bearer " + _generate_jwt(api_key, secret_key, auth_payload, expiration_time=6000))
    ]
    return list(metadata)


def _build_request(path: str, max_alternatives: int, do_not_perform_vad: bool, profanity_filter: bool):
    mp3_file = mp3.MP3(path)
    num_ch = int(mp3_file.info.channels)
    sr_audio = int(mp3_file.info.sample_rate)
    request = stt_pb2.RecognizeRequest()
    with open(path, "rb") as f:
        request.audio.content = f.read()

    request.config.encoding = stt_pb2.AudioEncoding.MPEG_AUDIO
    request.config.sample_rate_hertz = sr_audio
    request.config.num_channels = num_ch  # количество каналов в записи

    request.config.max_alternatives = max_alternatives  # включение альтернативных распознаваний
    request.config.do_not_perform_vad = do_not_perform_vad  # отключение режима диалога
    request.config.profanity_filter = profanity_filter  # фильтр ненормативной лексики
    return request


def tinkoff_api(api_key: str, secret_key: str, max_alternatives: int = 3, do_not_perform_vad: bool = True,
            profanity_filter: bool = True, expiration_time: int = int(6e4), endpoint: str = 'stt.tinkoff.ru:443'):
    stub = _SpeechToTextStub(grpc.secure_channel(endpoint, grpc.ssl_channel_credentials()))
    metadata = _authorization_metadata(api_key, secret_key, "tinkoff.cloud.stt", expiration_time)

    def fun(path):
        response = stub.Recognize(_build_request(
            path, max_alternatives, do_not_perform_vad, profanity_filter
        ), metadata=metadata)

        tinkoff_res = ''
        for result in response.results:
            if int(result.channel) == int(0):
                ch = '-  '
            elif int(result.channel) == int(1):
                ch = '-- '

            for alternative in result.alternatives:
                tinkoff_res += '\n' + ch + alternative.transcript

        return tinkoff_res

    return fun
