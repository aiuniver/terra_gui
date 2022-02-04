import json
import dotenv

from django.conf import settings
from rest_framework.exceptions import APIException

from apps.api import decorators, remote
from apps.api.base import BaseAPIView, BaseResponseSuccess
from apps.api.serializers import LoginSerializer
from apps.plugins.frontend import defaults_data


class NotFoundAPIView(BaseAPIView):
    def dispatch(self, request, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = self.default_response_headers
        response = self.handle_exception(
            APIException(f"API-метод {request.path} не найден")
        )
        self.response = self.finalize_response(request, response, *args, **kwargs)
        return self.response


class LoginAPIView(BaseAPIView):
    @decorators.serialize_data(LoginSerializer)
    def post(self, request, serializer, **kwargs):
        response = remote.request("/login/", serializer.validated_data)
        settings.USER_SESSION = response.get("session")
        if settings.USER_KEEP_SESSION:
            env = dotenv.dotenv_values(settings.ENV_FILE)
            env.update({"USER_SESSION": settings.USER_SESSION})
            with open(settings.ENV_FILE, "w") as env_file_ref:
                data = "\n".join(
                    list(map(lambda item: f"{item[0]}={item[1]}", env.items()))
                )
                env_file_ref.write(f"{data}\n")
        return BaseResponseSuccess({"url": "/datasets"})


class ConfigAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        return BaseResponseSuccess(
            {
                "defaults": json.loads(defaults_data.json()),
                "project": json.loads(request.project.frontend()),
                "user": settings.USER,
            }
        )
