from rest_framework.views import APIView
from rest_framework.response import Response

from apps.plugins.terra import terra_exchange
from .data.exchange import ExchangeData


class ExchangeAPIView(APIView):
    def post(self, request, *args, **kwargs):
        data = ExchangeData(name=kwargs.get("name"), data=request.data)
        return Response(data.__dict__)


class LayersTypesAPIView(APIView):
    def get(self, request, *args, **kwargs):
        term = request.GET.get("term").lower()
        available = list(
            filter(
                lambda item: item.lower().find(term) != -1,
                terra_exchange.project.layers_types.keys(),
            )
        )
        items = []
        for item in available:
            items.append({"id": item, "label": item, "value": item})
        return Response(items)
