from django.utils.deprecation import MiddlewareMixin

from . import terra_exchange


class TerraProjectMiddleware(MiddlewareMixin):
    def process_request(self, request):
        response = terra_exchange.call("get_state")

        if response.success:
            response.data.update({"error": ""})
            data = response.data
        else:
            data = {"error": "No connection to TerraAI project"}

        terra_exchange.project = data
        for dts in data["datasets"]:
            print(dts)
        request.terra_project = terra_exchange.project
