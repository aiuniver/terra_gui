from pydantic import ValidationError

from apps.plugins.project import data_path
from terra_ai.agent import agent_exchange

from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import ModelGetSerializer


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            model = agent_exchange("model_get", **serializer.validated_data)
            return BaseResponseSuccess(model.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = ModelGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            request.project.model = agent_exchange(
                "model_get", **serializer.validated_data
            )
            return BaseResponseSuccess(request.project.model.native())
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=str(data_path.modeling)).native()
        )


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            request.project.model = agent_exchange(
                "model_update", model=request.project.model.native(), **request.data
            )
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class LayerSaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            request.project.model = agent_exchange(
                "model_layer_save", model=request.project.model.native(), **request.data
            )
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)
