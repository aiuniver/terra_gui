import re
import sys

from rest_framework import serializers
from transliterate import slugify

from apps.plugins.frontend import choices as frontend_choices
from apps.api.fields import DirectoryPathField, DirectoryOrFilePathField

from terra_ai.settings import TERRA_PATH
from terra_ai.data.datasets.extra import LayerTypeChoice, LayerGroupChoice


class MinMaxScalerSerializer(serializers.Serializer):
    min_scaler: serializers.IntegerField(default=0)
    max_scaler: serializers.IntegerField(default=1)


class VersionsSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class ChoiceSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()
    version = serializers.CharField()
    reset_model = serializers.BooleanField(default=False)


class SourceLoadSerializer(serializers.Serializer):
    mode = serializers.CharField()
    value = serializers.CharField()
    architecture = serializers.CharField()


class LayerParametersSerializer(serializers.Serializer):
    sources_paths = serializers.ListSerializer(child=DirectoryOrFilePathField())

    def validate_sources_paths(self, value):
        if self.__class__ in [
            LayerParametersClassificationSerializer,
            LayerParametersTrackerSerializer,
        ]:
            return value
        if not len(value):
            raise serializers.ValidationError("Этот список не может быть пустым.")
        return value


class LayerParametersImageSerializer(MinMaxScalerSerializer, LayerParametersSerializer):
    width = serializers.IntegerField(min_value=1)
    height = serializers.IntegerField(min_value=1)
    net = serializers.ChoiceField(choices=frontend_choices.LayerNetChoice.items_tuple())
    scaler = serializers.ChoiceField(
        choices=frontend_choices.LayerScalerImageChoice.items_tuple()
    )


class LayerParametersTextSerializer(LayerParametersSerializer):
    filters = serializers.CharField(
        default='–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    )
    text_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerTextModeChoice.items_tuple()
    )
    max_words = serializers.IntegerField(required=False, min_value=1)
    length = serializers.IntegerField(required=False, min_value=1)
    step = serializers.IntegerField(required=False, min_value=1)
    pymorphy = serializers.BooleanField(default=False)
    prepare_method = serializers.ChoiceField(
        choices=frontend_choices.LayerPrepareMethodChoice.items_tuple()
    )
    max_words_count = serializers.IntegerField(required=False, min_value=1)
    word_to_vec_size = serializers.IntegerField(required=False, min_value=1)

    def __init__(self, instance=None, data=None, **kwargs):
        _text_mode = data.get("text_mode")
        if _text_mode == frontend_choices.LayerTextModeChoice.completely.name:
            self.fields.get("max_words").required = True
            data.pop("length", None)
            data.pop("step", None)
        elif _text_mode == frontend_choices.LayerTextModeChoice.length_and_step.name:
            self.fields.get("length").required = True
            self.fields.get("step").required = True
            data.pop("max_words", None)

        _prepare_method = data.get("prepare_method")
        if _prepare_method in [
            frontend_choices.LayerPrepareMethodChoice.embedding.name,
            frontend_choices.LayerPrepareMethodChoice.bag_of_words.name,
        ]:
            self.fields.get("max_words_count").required = True
            data.pop("word_to_vec_size", None)
        elif (
            _prepare_method
            == frontend_choices.LayerPrepareMethodChoice.word_to_vec.name
        ):
            self.fields.get("word_to_vec_size").required = True
            data.pop("max_words_count", None)

        super().__init__(instance=instance, data=data, **kwargs)


class LayerParametersAudioSerializer(MinMaxScalerSerializer, LayerParametersSerializer):
    sample_rate = serializers.IntegerField(min_value=1)
    audio_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerAudioModeChoice.items_tuple()
    )
    max_seconds = serializers.FloatField(required=False, min_value=0, allow_null=True)
    length = serializers.FloatField(required=False, min_value=0, allow_null=True)
    step = serializers.FloatField(required=False, min_value=0, allow_null=True)
    fill_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerAudioFillModeChoice.items_tuple()
    )
    parameter = serializers.ChoiceField(
        choices=frontend_choices.LayerAudioParameterChoice.items_tuple()
    )
    resample = serializers.ChoiceField(
        choices=frontend_choices.LayerAudioResampleChoice.items_tuple()
    )
    scaler = serializers.ChoiceField(
        choices=frontend_choices.LayerScalerAudioChoice.items_tuple()
    )

    def __init__(self, instance=None, data=None, **kwargs):
        _audio_mode = data.get("audio_mode")
        if _audio_mode == frontend_choices.LayerAudioModeChoice.completely.name:
            self.fields.get("max_seconds").required = True
            data.pop("length", None)
            data.pop("step", None)
        elif _audio_mode == frontend_choices.LayerAudioModeChoice.length_and_step.name:
            self.fields.get("length").required = True
            self.fields.get("step").required = True
            data.pop("max_seconds", None)

        super().__init__(instance=instance, data=data, **kwargs)


class LayerParametersVideoSerializer(MinMaxScalerSerializer, LayerParametersSerializer):
    width = serializers.IntegerField(min_value=1)
    height = serializers.IntegerField(min_value=1)
    fill_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerVideoFillModeChoice.items_tuple()
    )
    frame_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerVideoFrameModeChoice.items_tuple()
    )
    video_mode = serializers.ChoiceField(
        choices=frontend_choices.LayerVideoModeChoice.items_tuple()
    )
    max_frames = serializers.IntegerField(required=False, min_value=1)
    length = serializers.IntegerField(required=False, min_value=1)
    step = serializers.IntegerField(required=False, min_value=1)
    scaler = serializers.ChoiceField(
        choices=frontend_choices.LayerScalerVideoChoice.items_tuple()
    )

    def __init__(self, instance=None, data=None, **kwargs):
        _video_mode = data.get("video_mode")
        if _video_mode == frontend_choices.LayerAudioModeChoice.completely.name:
            self.fields.get("max_frames").required = True
            data.pop("length", None)
            data.pop("step", None)
        elif _video_mode == frontend_choices.LayerAudioModeChoice.length_and_step.name:
            self.fields.get("length").required = True
            self.fields.get("step").required = True
            data.pop("max_frames", None)

        super().__init__(instance=instance, data=data, **kwargs)


class LayerParametersDataframeSerializer(LayerParametersSerializer):
    cols_names = serializers.DictField(
        child=serializers.ListField(child=serializers.IntegerField()), default=dict
    )


class LayerParametersClassificationSerializer(LayerParametersSerializer):
    pass


class LayerParametersTrackerSerializer(LayerParametersSerializer):
    pass


class LayerParametersSpeech2TextSerializer(serializers.Serializer):
    pass


class LayerParametersText2SpeechSerializer(serializers.Serializer):
    pass


class LayerParametersImageGANSerializer(serializers.Serializer):
    pass


class LayerParametersImageCGANSerializer(serializers.Serializer):
    pass


class LayerParametersSegmentationSerializer(LayerParametersSerializer):
    width: serializers.IntegerField(min_value=1)
    height: serializers.IntegerField(min_value=1)
    mask_range = serializers.IntegerField(min_value=0)
    classes_names: serializers.ListSerializer(child=serializers.CharField())
    classes_colors: serializers.ListSerializer(child=serializers.CharField())


class LayerParametersTextSegmentationSerializer(serializers.Serializer):
    open_tags: serializers.CharField(required=False)
    close_tags: serializers.CharField(required=False)


class LayerParametersRegressionSerializer(
    MinMaxScalerSerializer, LayerParametersSerializer
):
    scaler: frontend_choices.LayerScalerRegressionChoice


class LayerParametersTimeseriesSerializer(
    MinMaxScalerSerializer, LayerParametersSerializer
):
    length: serializers.IntegerField(min_value=1)
    step: serializers.IntegerField(min_value=1)
    trend: serializers.BooleanField()
    trend_limit: serializers.CharField(required=False)
    depth: serializers.IntegerField(required=False, min_value=1)
    scaler: frontend_choices.LayerScalerTimeseriesChoice

    def __init__(self, instance=None, data=None, **kwargs):
        _trend = data.get("trend")
        if _trend:
            self.fields.get("trend_limit").required = True
        else:
            self.fields.get("depth").required = True
            self.fields.get("scaler").required = True

        super().__init__(instance=instance, data=data, **kwargs)


class LayerParametersObjectDetectionSerializer(LayerParametersSerializer):
    yolo = serializers.ChoiceField(
        choices=frontend_choices.LayerYoloVersionChoice.items_tuple()
    )


class CreateLayerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    parameters = serializers.DictField(required=False)


class CreateLayerInputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(
        choices=frontend_choices.LayerInputTypeChoice.items_tuple()
    )


class CreateLayerOutputSerializer(CreateLayerSerializer):
    type = serializers.ChoiceField(
        choices=frontend_choices.LayerOutputTypeChoice.items_tuple()
    )

    def __init__(self, *args, **kwargs):
        if kwargs.get("data", {}).get("type") == "TrackerImages":
            kwargs["data"]["type"] = "Tracker"
        super().__init__(*args, **kwargs)


class CreateTagSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()

    def get_alias(self, data):
        return re.sub(r"([\-]+)", "_", slugify(data.get("name"), language_code="ru"))


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


class CreateColumnProcessingSerializer(serializers.Serializer):
    type = serializers.ChoiceField(
        choices=frontend_choices.ColumnProcessingInputTypeChoice.items_tuple()
        + frontend_choices.ColumnProcessingOutputTypeChoice.items_tuple()
    )
    parameters = serializers.DictField()


class CreateSerializer(serializers.Serializer):
    alias = serializers.SerializerMethodField()
    name = serializers.CharField()
    datasets_path = DirectoryPathField(default=str(TERRA_PATH.datasets.absolute()))
    source_path = DirectoryPathField()
    info = CreateInfoSerializer()
    tags = serializers.ListSerializer(child=CreateTagSerializer(), default=[])
    use_generator = serializers.BooleanField(default=False)
    columns_processing = serializers.DictField(
        child=CreateColumnProcessingSerializer(), default=dict
    )
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


class CreateValidateBlockSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
    name = serializers.CharField()
    type = serializers.ChoiceField(choices=tuple(LayerTypeChoice.values()))
    removable = serializers.BooleanField(default=False)
    bind = LayerBindSerializer()
    position = serializers.ListSerializer(child=serializers.IntegerField())
    parameters = serializers.DictField()


class CreateValidateSerializer(serializers.Serializer):
    type = serializers.ChoiceField(choices=tuple(LayerGroupChoice.values()))
    items = CreateValidateBlockSerializer(many=True)


class DeleteSerializer(serializers.Serializer):
    group = serializers.CharField()
    alias = serializers.CharField()


class SourceSegmentationClassesAutosearchSerializer(serializers.Serializer):
    num_classes = serializers.IntegerField(min_value=1)
    mask_range = serializers.IntegerField(min_value=0)
