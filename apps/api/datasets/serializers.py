import re

from django.core.exceptions import ValidationError

from rest_framework import serializers
from transliterate import slugify

from apps.plugins.project import data_path
from apps.plugins.frontend.choices import LayerInputTypeChoice, LayerOutputTypeChoice

from ..fields import DirectoryPathField


class ChoiceSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class SourceLoadSerializer(serializers.Serializer):
    mode = serializers.CharField()
    value = serializers.CharField()


class CreateLayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    parameters = serializers.Serializer()

    def validate(self, attrs):
        _type = attrs.get("type")
        # print(attrs.get("parameters"))
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
            raise ValidationError({"part": "Сумма значений должна быть равной 1"})
        return super().validate(attrs)


class CreateSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    datasets_path = DirectoryPathField(default=str(data_path.datasets.absolute()))
    source_path = DirectoryPathField()
    info = CreateInfoSerializer()
    tags = serializers.ListSerializer(child=CreateTagSerializer())
    use_generator = serializers.BooleanField(default=False)
    inputs = serializers.ListSerializer(child=CreateInputSerializer())
    outputs = serializers.ListSerializer(child=CreateOutputSerializer())

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))
