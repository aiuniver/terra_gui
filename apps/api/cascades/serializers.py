from rest_framework import serializers

from terra_ai.data.cascades.extra import BlockGroupChoice


class CascadeGetSerializer(serializers.Serializer):
    value = serializers.CharField()


class BlockBindSerializer(serializers.Serializer):
    up = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))
    down = serializers.ListSerializer(child=serializers.IntegerField(min_value=1))


class BlockSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    group = serializers.ChoiceField(choices=tuple(BlockGroupChoice.values()))
    bind = BlockBindSerializer()
    position = serializers.ListSerializer(child=serializers.FloatField())
    parameters = serializers.DictField() 

    
class UpdateSerializer(serializers.Serializer):
    blocks = serializers.ListSerializer(child=BlockSerializer())
