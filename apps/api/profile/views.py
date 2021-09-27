from terra_ai.agent import agent_exchange
from terra_ai.agent.exceptions import ExchangeBaseException

from ..base import (
    BaseAPIView,
    BaseResponseSuccess,
    BaseResponseErrorFields,
    BaseResponseErrorGeneral,
)
from .serializers import SaveSerializer


class SaveAPIView(BaseAPIView):
    def post(self, request):
        serializer = SaveSerializer(data=request.data)
        if not serializer.is_valid():
            return BaseResponseErrorFields(serializer.errors)
        try:
            agent_exchange("profile_save", data=serializer.validated_data)
            return BaseResponseSuccess()
        except ExchangeBaseException as error:
            return BaseResponseErrorGeneral(str(error))
