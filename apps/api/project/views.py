from ..base import BaseAPIView, BaseResponseSuccess, BaseResponseErrorFields
from .serializers import NameSerializer


class NameAPIView(BaseAPIView):
    def post(self, request, **kwargs):
        serializer = NameSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        request.project.name = serializer.validated_data.get("name")
        return BaseResponseSuccess()
