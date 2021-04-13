import json
import requests

from django.conf import settings

from .data import TerraExchangeResponse, TerraExchangeProject
from .exceptions import TerraExchangeException


class TerraExchange:
    _project: TerraExchangeProject = TerraExchangeProject()

    @property
    def project(self) -> TerraExchangeProject:
        return self._project

    @property
    def api_url(self) -> str:
        return settings.TERRA_EXCHANGE_API_URL

    def __get_api_url(self, name: str) -> str:
        return f"{self.api_url}/{name}/"

    def __request_get(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«__request_get» method must contain method name as first argument"
            )
        try:
            response = requests.get(self.__get_api_url(args[0]))
            return self.__response(response)
        except requests.exceptions.ConnectionError as error:
            return TerraExchangeResponse(success=False, error=error)
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=error)

    def __request_post(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«__request_post» method must contain method name as first argument"
            )
        try:
            response = requests.post(self.__get_api_url(args[0]), json=kwargs)
            return self.__response(response)
        except requests.exceptions.ConnectionError as error:
            return TerraExchangeResponse(success=False, error=error)
        except json.JSONDecodeError as error:
            return TerraExchangeResponse(success=False, error=error)

    def __response(self, response: requests.models.Response) -> TerraExchangeResponse:
        if response.ok:
            return TerraExchangeResponse(**response.json())
        else:
            return TerraExchangeResponse(
                success=False, error=response.json().get("detail")
            )

    def call(self, *args, **kwargs) -> TerraExchangeResponse:
        if len(args) != 1:
            raise TerraExchangeException(
                "«call» method must contain method name as first argument"
            )

        name = args[0]
        method = getattr(self, f"_call_{name}", None)
        if method:
            return method(**kwargs)
        else:
            raise TerraExchangeException(f"You call undefined method «{name}»")

    def _call_get_state(self):
        return self.__request_post("get_state")

    def _call_set_project_name(self, name: str):
        self._project.name = name
        return TerraExchangeResponse()

    def _call_prepare_dataset(self, dataset: str, task: str):
        response = self.__request_post(
            "prepare_dataset", dataset_name=dataset, task_type=task
        )
        if response.success:
            self._project.dataset = dataset
            self._project.task = task
            response.data = {"dataset": dataset, "task": task}
        return response

    def _call_get_data(self):
        return self.__request_post("get_data")
