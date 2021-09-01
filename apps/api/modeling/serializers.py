from rest_framework import serializers

from terra_ai.data.modeling.layers import Layer
from terra_ai.data.modeling.extra import LayerGroupChoice


class ModelGetSerializer(serializers.Serializer):
    value = serializers.CharField(required=True)


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
    type = serializers.ChoiceField(choices=tuple(map(lambda item: item.name, Layer)))
    group = serializers.ChoiceField(choices=tuple(LayerGroupChoice.values()))
    bind = LayerBindSerializer()
    shape = LayerShapeSerializer()
    position = serializers.ListSerializer(child=serializers.IntegerField())
    parameters = serializers.DictField()


class UpdateSerializer(serializers.Serializer):
    layers = serializers.ListSerializer(child=LayerSerializer())
