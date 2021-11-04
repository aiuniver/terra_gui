from rest_framework import serializers

from terra_ai.data.modeling.extra import LayerTypeChoice, LayerGroupChoice


class ModelGetSerializer(serializers.Serializer):
    value = serializers.CharField(required=True)
    reset_dataset = serializers.BooleanField(default=False)


class LayerBindSerializer(serializers.Serializer):
    up = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))
    down = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))


class LayerShapeSerializer(serializers.Serializer):
    input = serializers.ListSerializer(
        child=serializers.ListSerializer(child=serializers.IntegerField(min_value=1))
    )


class LayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    type = serializers.ChoiceField(
        choices=tuple(map(lambda item: (item.value, item.name), LayerTypeChoice))
    )
    group = serializers.ChoiceField(choices=tuple(LayerGroupChoice.values()))
    bind = LayerBindSerializer()
    shape = LayerShapeSerializer(required=False)
    task = serializers.CharField(required=False, allow_null=True)
    num_classes = serializers.IntegerField(required=False, min_value=1, allow_null=True)
    position = serializers.ListSerializer(child=serializers.FloatField())
    parameters = serializers.DictField()


class UpdateSerializer(serializers.Serializer):
    layers = serializers.ListSerializer(child=LayerSerializer())


class PreviewSerializer(serializers.Serializer):
    preview = serializers.CharField()


class CreateSerializer(serializers.Serializer):
    name = serializers.CharField()
    overwrite = serializers.BooleanField(default=False)
    preview = serializers.CharField()


class DatatypeSerializer(serializers.Serializer):
    source = serializers.IntegerField(min_value=1)
    target = serializers.IntegerField(min_value=1)
