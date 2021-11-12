import base64

from tempfile import NamedTemporaryFile

from apps.api import utils
from apps.api.cascades.serializers import (
    CascadeGetSerializer,
    UpdateSerializer,
    PreviewSerializer,
)
from apps.plugins.project import project_path
from terra_ai.agent import agent_exchange

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
)


class GetAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class InfoAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            agent_exchange("cascades_info", path=project_path.cascades).native()
        )


class LoadAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = CascadeGetSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = agent_exchange(
            "cascade_get", value=serializer.validated_data.get("value")
        )
        return BaseResponseSuccess(cascade.native())


class UpdateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = UpdateSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        cascade = request.project.cascade
        data = serializer.validated_data
        cascade_data = cascade.native()
        cascade_data.update(data)
        cascade = agent_exchange("cascade_update", cascade=cascade_data)
        request.project.set_cascade(cascade)
        return BaseResponseSuccess({"blocks": cascade.blocks.native()})


class ClearAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        request.project.clear_cascade()
        return BaseResponseSuccess(request.project.cascade.native())


class ValidateAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("cascade_validate", cascade=request.project.cascade)
        return BaseResponseSuccess()


class StartAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        agent_exchange("cascade_start", cascade=request.project.cascade)
        return BaseResponseSuccess()


class SaveAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        # agent_exchange("cascade_start", cascade=request.project.cascade)
        return BaseResponseSuccess()


class PreviewAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = PreviewSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        filepath = NamedTemporaryFile(suffix=".png")  # Add for Win ,delete=False
        filepath.write(base64.b64decode(serializer.validated_data.get("preview")))
        utils.autocrop_image_square(filepath.name, min_size=600)
        with open(filepath.name, "rb") as filepath_ref:
            content = filepath_ref.read()
            return BaseResponseSuccess(base64.b64encode(content))
