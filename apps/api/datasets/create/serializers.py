import re

from transliterate import slugify
from rest_framework import serializers

from terra_ai.data.datasets.extra import (
    LayerTypeChoice,
    LayerGroupChoice,
    DatasetGroupChoice,
    SourceModeChoice,
)
from terra_ai.data.training.extra import ArchitectureChoice


class VersionSerializer(serializers.Serializer):
    group = serializers.ChoiceField(choices=DatasetGroupChoice.values())
    alias = serializers.CharField()
    version = serializers.CharField(required=False, allow_blank=True)


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


class CreateSourceSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=SourceModeChoice.values())
    value = serializers.CharField()
    path = serializers.CharField()


class CreateVersionSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    info = CreateInfoSerializer()
    inputs = serializers.ListSerializer(child=serializers.DictField())
    outputs = serializers.ListSerializer(child=serializers.DictField())

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))


class CreateSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    source = CreateSourceSerializer()
    architecture = serializers.ChoiceField(choices=ArchitectureChoice.values())
    tags = serializers.ListSerializer(child=serializers.CharField(), default=[])
    version = CreateVersionSerializer()

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))


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


class ValidateGroupsSerializer(serializers.Serializer):
    inputs = ValidateBlockSerializer(many=True, default=[])
    outputs = ValidateBlockSerializer(many=True, default=[])


class ValidateSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=tuple(LayerGroupChoice.values()))
    architecture = serializers.ChoiceField(choices=tuple(ArchitectureChoice.values()))
    items = ValidateGroupsSerializer()
