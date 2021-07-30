from rest_framework import serializers
from transliterate import slugify


class UploadSerializer(serializers.Serializer):
    name = serializers.CharField()
    url = serializers.CharField(required=False, default="")
    replace = serializers.BooleanField(default=False)

    def validate(self, attrs):
        attrs.update({"url": slugify(attrs.get("name"))})
        return super().validate(attrs)
