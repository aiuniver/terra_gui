from rest_framework import serializers


class AutosearchSerializer(serializers.Serializer):
    num_classes = serializers.IntegerField(min_value=1)
    mask_range = serializers.IntegerField(min_value=0)
