import re
import sys

from rest_framework import serializers
from transliterate import slugify

from apps.plugins.project import data_path
from apps.plugins.frontend.choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
    LayerAudioModeChoice,
    LayerAudioParameterChoice,
)

from ..fields import DirectoryPathField, DirectoryOrFilePathField


class ChoiceSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class SourceLoadSerializer(serializers.Serializer):
    mode = serializers.CharField()
    value = serializers.CharField()


class LayerParametersSerializer(serializers.Serializer):
    sources_paths = serializers.ListSerializer(child=DirectoryOrFilePathField())

    def validate_sources_paths(self, value):
        if not len(value):
            raise serializers.ValidationError("Этот список не может быть пустым.")
        return value


class LayerParametersImageSerializer(LayerParametersSerializer):
    width = serializers.IntegerField(min_value=1)
    height = serializers.IntegerField(min_value=1)
    net = serializers.ChoiceField(choices=LayerNetChoice.items_tuple())
    scaler = serializers.ChoiceField(choices=LayerScalerChoice.items_tuple())


class LayerParametersTextSerializer(LayerParametersSerializer):
    max_words_count = serializers.IntegerField(min_value=1)
    delete_symbols = serializers.CharField(default="", allow_blank=True)
    text_mode = serializers.ChoiceField(choices=LayerTextModeChoice.items_tuple())
    max_words = serializers.IntegerField(required=False, min_value=1)
    length = serializers.IntegerField(required=False, min_value=1)
    step = serializers.IntegerField(required=False, min_value=1)
    pymorphy = serializers.BooleanField(default=False)
    embedding = serializers.BooleanField(default=False)
    bag_of_words = serializers.BooleanField(default=False)
    word_to_vec = serializers.BooleanField(default=False)
    prepare_method = serializers.ChoiceField(
        choices=LayerPrepareMethodChoice.items_tuple()
    )
    word_to_vec_size = serializers.IntegerField(required=False, min_value=1)

    def __init__(self, instance=None, data=None, **kwargs):
        _text_mode = data.get("text_mode")
        if _text_mode == LayerTextModeChoice.completely.name:
            self.fields.get("max_words").required = True
        elif _text_mode == LayerTextModeChoice.length_and_step.name:
            self.fields.get("length").required = True
            self.fields.get("step").required = True

        _prepare_method = data.get("prepare_method")
        if _prepare_method == LayerPrepareMethodChoice.word_to_vec.name:
            self.fields.get("word_to_vec_size").required = True
        data.update({_prepare_method: True})

        super().__init__(instance=instance, data=data, **kwargs)


class LayerParametersAudioSerializer(LayerParametersSerializer):
    sample_rate = serializers.IntegerField(min_value=1)
    audio_mode = serializers.ChoiceField(choices=LayerAudioModeChoice.items_tuple())
    max_seconds = serializers.IntegerField(required=False, min_value=1)
    length = serializers.IntegerField(required=False, min_value=1)
    step = serializers.IntegerField(required=False, min_value=1)
    parameter = serializers.ChoiceField(choices=LayerAudioParameterChoice.items_tuple())

    def __init__(self, instance=None, data=None, **kwargs):
        _audio_mode = data.get("audio_mode")
        if _audio_mode == LayerAudioModeChoice.completely.name:
            self.fields.get("max_seconds").required = True
        elif _audio_mode == LayerAudioModeChoice.length_and_step.name:
            self.fields.get("length").required = True
            self.fields.get("step").required = True

        super().__init__(instance=instance, data=data, **kwargs)


class CreateLayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    parameters = serializers.DictField(required=False)


class CreateLayerInputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(choices=LayerInputTypeChoice.items_tuple())


class CreateLayerOutputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(choices=LayerOutputTypeChoice.items_tuple())


class CreateTagSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))


class CreateInfoPartSerializer(serializers.Serializer):
    test = serializers.FloatField(min_value=0.05, max_value=0.9)
    train = serializers.FloatField(min_value=0.05, max_value=0.9)
    validation = serializers.FloatField(min_value=0.05, max_value=0.9)


class CreateInfoSerializer(serializers.Serializer):
    shuffle = serializers.BooleanField(default=True)
    part = CreateInfoPartSerializer()

    def validate(self, attrs):
        _part = attrs.get("part")
        _test = _part.get("test")
        _train = _part.get("train")
        _validation = _part.get("validation")
        if _test + _train + _validation != 1.0:
            raise serializers.ValidationError(
                {"part": "Сумма значений должна быть равной 1"}
            )
        return super().validate(attrs)


class CreateSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    datasets_path = DirectoryPathField(default=str(data_path.datasets.absolute()))
    source_path = DirectoryPathField()
    info = CreateInfoSerializer()
    tags = serializers.ListSerializer(child=CreateTagSerializer(), default=[])
    use_generator = serializers.BooleanField(default=False)
    inputs = serializers.ListSerializer(child=serializers.DictField())
    outputs = serializers.ListSerializer(child=serializers.DictField())

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))

    def _validate_layer(self, create_class, value) -> dict:
        _errors = {}
        _serializer = create_class(data=value)
        if not _serializer.is_valid():
            _errors.update(**_serializer.errors)
        _id = value.get("id", 0)
        _type = value.get("type")
        if _id and _type:
            _classname = f"LayerParameters{_type}Serializer"
            _serializer_class = getattr(
                sys.modules.get(__name__, None),
                f"LayerParameters{_type}Serializer",
                None,
            )
            if _serializer_class:
                _serializer_parameters = _serializer_class(
                    data=value.get("parameters", {})
                )
                if not _serializer_parameters.is_valid():
                    _errors.update({"parameters": _serializer_parameters.errors})
            else:
                _errors.update({"parameters": ["Нет класса для обработки параметров"]})
        return _errors

    def validate_inputs(self, value: list) -> list:
        _errors = {}
        if not len(value):
            raise serializers.ValidationError("Этот список не может быть пустым.")
        for item in value:
            _error = self._validate_layer(CreateLayerInputSerializer, item)
            if len(_error.keys()):
                _errors.update({item.get("id", 0): _error})
        if _errors:
            raise serializers.ValidationError(_errors)
        return value

    def validate_outputs(self, value: list) -> list:
        _errors = {}
        if not len(value):
            raise serializers.ValidationError("Этот список не может быть пустым.")
        for item in value:
            _error = self._validate_layer(CreateLayerOutputSerializer, item)
            if len(_error.keys()):
                _errors.update({item.get("id", 0): _error})
        if _errors:
            raise serializers.ValidationError(_errors)
        return value
