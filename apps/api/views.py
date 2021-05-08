from rest_framework.views import APIView
from rest_framework.response import Response

from .data.exchange import ExchangeData


class ExchangeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        data = ExchangeData(name=kwargs.get("name"), data=request.data)
        return Response(data.__dict__)
