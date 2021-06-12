from rest_framework import serializers
from rest_framework.exceptions import ValidationError


class DatasetSourceNumLinksSerializers(serializers.Serializer):
    inputs = serializers.IntegerField(required=True, min_value=1)
    outputs = serializers.IntegerField(required=True, min_value=1)


class DatasetSourceSerializers(serializers.Serializer):
    MODE_GOOGLE_DRIVE = "google_drive"
    MODE_URL = "url"
    MODE_CHOICES = ((MODE_GOOGLE_DRIVE, "Google drive"), (MODE_URL, "URL-ссылка"))

    mode = serializers.ChoiceField(choices=MODE_CHOICES)
    name = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    link = serializers.URLField(required=False, allow_blank=True, allow_null=True)
    num_links = DatasetSourceNumLinksSerializers(required=True)

    def validate(self, attrs):
        attrs = super().validate(attrs)
        mode = attrs.get("mode")
        if mode == self.MODE_GOOGLE_DRIVE and not attrs.get("name"):
            raise ValidationError({"name": "Выберите zip-файл Google-диска"})
        if mode == self.MODE_URL and not attrs.get("link"):
            raise ValidationError({"link": "Введите URL на zip-файл"})
        return attrs


class DatasetCreateDictParametersSerializer(serializers.Serializer):
    name = serializers.CharField()
    preserve_sequence = serializers.BooleanField()
    test_part = serializers.FloatField(required=True, min_value=0, max_value=1)
    train_part = serializers.FloatField(required=True, min_value=0, max_value=1)
    val_part = serializers.FloatField(required=True, min_value=0, max_value=1)
    user_tags: serializers.CharField(required=False, allow_blank=True, allow_null=True)


class DatasetCreateDictSerializer(serializers.Serializer):
    inputs = serializers.DictField()
    outputs = serializers.DictField()
    parameters = DatasetCreateDictParametersSerializer()


class DatasetCreateSerializer(serializers.Serializer):
    dataset_dict = DatasetCreateDictSerializer()
