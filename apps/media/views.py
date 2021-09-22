import mimetypes

from django.views import View
from django.http.response import HttpResponse, HttpResponseNotFound

from .serializers import RequestFileDataSerializer


class BaseMixinView(View):
    def get(self, request, *args, **kwargs):
        serializer = RequestFileDataSerializer(data=request.GET)
        if not serializer.is_valid():
            return HttpResponseNotFound()

        path = serializer.data.get("path")
        with open(path, "rb") as path_ref:
            response_data = {"content": path_ref.read()}
            path_mimetype = mimetypes.guess_type(path)
            if path_mimetype:
                response_data.update({"content_type": path_mimetype[0]})
            return HttpResponse(**response_data)


class BlankView(BaseMixinView):
    pass
