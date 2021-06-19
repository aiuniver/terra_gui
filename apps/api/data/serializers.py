from rest_framework import serializers

from .validators import validate_restriction_name


class SetProjectNameSerializer(serializers.Serializer):
    name = serializers.CharField()


class PrepareDatasetSerializer(serializers.Serializer):
    dataset = serializers.CharField()
    is_custom = serializers.BooleanField(required=False, default=False)


class GetAutoColorsSerializer(serializers.Serializer):
    name = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    num_classes = serializers.IntegerField(required=False, allow_null=True, min_value=1)
    mask_range = serializers.IntegerField(min_value=1)
    txt_file = serializers.BooleanField(required=False, default=False)


class GetModelFromListSerializer(serializers.Serializer):
    model_file = serializers.CharField()
    is_terra = serializers.BooleanField(required=False, default=True)


class LayerSerializer(serializers.Serializer):
    x = serializers.FloatField(required=False, allow_null=True)
    y = serializers.FloatField(required=False, allow_null=True)
    down_link = serializers.ListSerializer(
        required=True,
        allow_null=False,
        allow_empty=True,
        child=serializers.IntegerField(min_value=1),
    )
    config = serializers.DictField()


class SetModelSerializer(serializers.Serializer):
    layers = serializers.DictField(child=LayerSerializer())
    reset_training = serializers.BooleanField(required=False, default=False)
    schema = serializers.ListSerializer(
        required=False,
        default=[],
        child=serializers.ListSerializer(
            required=False,
            allow_null=False,
            allow_empty=False,
            child=serializers.IntegerField(allow_null=True, min_value=1),
        ),
    )


class SaveModelSerializer(serializers.Serializer):
    name = serializers.CharField(validators=[validate_restriction_name])
    preview = serializers.CharField()
    overwrite = serializers.BooleanField(required=False, default=False)


class SaveLayerSerializer(serializers.Serializer):
    index = serializers.IntegerField(min_value=1)
    layer = LayerSerializer()


class GetChangeValidationSerializer(serializers.Serializer):
    index = serializers.IntegerField(min_value=1)
    layer = LayerSerializer()


class OptimizerParamsSerializer(serializers.Serializer):
    main = serializers.DictField(required=False, default={})
    extra = serializers.DictField(required=False, default={})


class OptimizerSerializer(serializers.Serializer):
    name = serializers.CharField()
    params = OptimizerParamsSerializer()


class BeforeStartTrainingSerializer(serializers.Serializer):
    batch_sizes = serializers.IntegerField(min_value=1)
    epochs_count = serializers.IntegerField(min_value=1)
    checkpoint = serializers.DictField(required=False, default={})
    optimizer = OptimizerSerializer()
    outputs = serializers.DictField(required=False, default={})


class ProjectSaveSerializer(serializers.Serializer):
    name = serializers.CharField(validators=[validate_restriction_name])
    overwrite = serializers.BooleanField(required=False, default=False)


class GetProjectSerializer(serializers.Serializer):
    name = serializers.CharField()
