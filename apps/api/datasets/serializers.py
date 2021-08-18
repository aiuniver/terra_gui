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

    # def run_validation(self, data):
    #     self.max_words.required = True
    #     return super().run_validation(data)

    # def validate(self, attrs):
    #     _errors = {}
    #     text_mode = attrs.get("text_mode")
    #     if text_mode == LayerTextModeChoice.completely.name:
    #         if not attrs.get("max_words"):
    #             _errors.update(
    #                 {"max_words": self.get_fields().get("max_words").fail("required")}
    #             )
    #
    #     elif text_mode == LayerTextModeChoice.length_and_step.name:
    #         self.get_fields().get("max_words").required = True
    #     if len(_errors.keys()):
    #         raise ValidationError(_errors)
    #     return super().validate(attrs)


class CreateLayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    parameters = serializers.DictField()

    def validate(self, attrs):
        _type = attrs.get("type")
        _classname = f"LayerParameters{_type}Serializer"
        _serializer_class = getattr(
            sys.modules.get(__name__, None), f"LayerParameters{_type}Serializer", None
        )
        if _serializer_class:
            _serializer = _serializer_class(data=attrs.get("parameters"))
            _serializer.is_valid(raise_exception=True)
            _parameters = _serializer.data
        else:
            _parameters = {}
        attrs.update({"parameters": _parameters})
        return super().validate(attrs)


class CreateInputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(choices=LayerInputTypeChoice.items_tuple())


class CreateOutputSerializer(CreateLayerSerializer):
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
    inputs = serializers.ListSerializer(child=CreateInputSerializer())
    outputs = serializers.ListSerializer(child=CreateOutputSerializer())

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))
