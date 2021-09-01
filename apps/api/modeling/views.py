import re

from pydantic import ValidationError
from transliterate import slugify

from apps.plugins.project import data_path
from apps.plugins.project import exceptions as project_exceptions

from terra_ai.agent import agent_exchange
from terra_ai.agent import exceptions as agent_exceptions
from terra_ai.data.modeling.model import ModelDetailsData

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import ModelGetSerializer, UpdateSerializer, CreateSerializer


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
            request.project.set_model(
                agent_exchange("model_get", **serializer.validated_data)
            )
            return BaseResponseSuccess(request.project.model.native())
        except project_exceptions.ProjectException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("models", path=str(data_path.modeling)).native()
        )


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_model()
        return BaseResponseSuccess()


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            request.project.set_model(
                agent_exchange(
                    "model_update",
                    model=request.project.model.native(),
                    **serializer.validated_data,
                )
            )
            return BaseResponseSuccess()
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class ValidateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        try:
            model, errors = agent_exchange(
                "model_validate", model=request.project.model
            )
            request.project.set_model(model)
            return BaseResponseSuccess(errors)
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class CreateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CreateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            model_data = request.project.model.native()
            model_data.update(
                {
                    "name": serializer.validated_data.get("name"),
                    "alias": re.sub(
                        r"([\-]+)",
                        "_",
                        slugify(
                            serializer.validated_data.get("name"), language_code="ru"
                        ),
                    ),
                    "image": serializer.validated_data.get("preview"),
                }
            )
            model = ModelDetailsData(**model_data)
            return BaseResponseSuccess(
                data=agent_exchange(
                    "model_create",
                    model=model.native(),
                    path=str(data_path.modeling),
                    overwrite=serializer.validated_data.get("overwrite"),
                )
            )
        except agent_exceptions.ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
        except ValidationError as error:
            return BaseResponseErrorFields(error)


class DeleteAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess()
