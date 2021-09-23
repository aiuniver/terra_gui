import mimetypes

from pathlib import Path

from django.views import View
from django.http.response import HttpResponse, HttpResponseNotFound

from .serializers import RequestFileDataSerializer
from .utils import path_hash


class BaseMixinView(View):
    def get(self, request, *args, **kwargs):
        serializer = RequestFileDataSerializer(data=request.GET)
        if not serializer.is_valid():
            return HttpResponseNotFound()

        hashstr = serializer.data.get("hash")
        path = path_hash(hashstr=hashstr)
        if not path:
            return HttpResponseNotFound()

        path = Path(path)
        with open(path, "rb") as path_ref:
            response_data = {"content": path_ref.read()}
            path_mimetype = mimetypes.guess_type(path)
            if path_mimetype:
                response_data.update({"content_type": path_mimetype[0]})
            return HttpResponse(**response_data)


class BlankView(BaseMixinView):
    pass
