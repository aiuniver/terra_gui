from rest_framework.views import APIView
from rest_framework.response import Response

from .data.exchange import ExchangeData


class ExchangeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        print("-------------------------------")
        print(request.headers.keys())
        print(request.headers.get("X-Colab-State"))
        print("-------------------------------")
        data = ExchangeData(name=kwargs.get("name"), data=request.data)
        return Response(data.__dict__)
