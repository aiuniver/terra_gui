import re
import sys

from transliterate import slugify
from rest_framework import serializers

from terra_ai.settings import TERRA_PATH
from terra_ai.data.datasets.extra import LayerTypeChoice, LayerGroupChoice
from terra_ai.data.training.extra import ArchitectureChoice

from apps.api.fields import DirectoryPathField
from apps.plugins.frontend.choices import LayerInputTypeChoice, LayerOutputTypeChoice


class CreateLayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    parameters = serializers.DictField(required=False)


class CreateLayerInputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(choices=LayerInputTypeChoice.items_tuple())


class CreateLayerOutputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(choices=LayerOutputTypeChoice.items_tuple())

    def __init__(self, *args, **kwargs):
        if kwargs.get("data", {}).get("type") == "TrackerImages":
            kwargs["data"]["type"] = "Tracker"
        super().__init__(*args, **kwargs)


class CreateInfoPartSerializer(serializers.Serializer):
    train = serializers.FloatField(min_value=0.1, max_value=0.9)
    validation = serializers.FloatField(min_value=0.1, max_value=0.9)


class CreateInfoSerializer(serializers.Serializer):
    shuffle = serializers.BooleanField(default=True)
    part = CreateInfoPartSerializer()

    def validate(self, attrs):
        _part = attrs.get("part")
        _train = _part.get("train")
        _validation = _part.get("validation")
        if _train + _validation != 1.0:
            raise serializers.ValidationError(
                {"part": "Сумма значений должна быть равной 1"}
            )
        return super().validate(attrs)


class CreateSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    datasets_path = DirectoryPathField(default=str(TERRA_PATH.datasets.absolute()))
    source_path = DirectoryPathField()
    info = CreateInfoSerializer()
    tags = serializers.ListSerializer(child=serializers.CharField(), default=[])
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


class LayerBindSerializer(serializers.Serializer):
    up = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))
    down = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))


class ValidateBlockSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    type = serializers.ChoiceField(choices=tuple(LayerTypeChoice.values()))
    removable = serializers.BooleanField(default=False)
    bind = LayerBindSerializer()
    position = serializers.ListSerializer(child=serializers.IntegerField())
    parameters = serializers.DictField()


class ValidateSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=tuple(LayerGroupChoice.values()))
    architecture = serializers.ChoiceField(choices=tuple(ArchitectureChoice.values()))
    items = ValidateBlockSerializer(many=True)
