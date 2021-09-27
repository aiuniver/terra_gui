import requests

from django.conf import settings

from ..agent import exceptions as agent_exceptions


def update_user_data(first_name, last_name) -> None:
    url = "http://terra.neural-university.ru/api/v1/update/"
    data = {
        "email": settings.USER_EMAIL,
        "user_token": settings.USER_TOKEN,
        "first_name": first_name,
        "last_name": last_name
    }
    response = requests.post(url, json=data)
    if (requests.status_codes.codes.get("ok") != response.status_code) or response.json().get('success'):
        raise agent_exceptions.FailedUpdateProfileException()


def update_user_token() -> str:
    url = "http://terra.neural-university.ru/api/v1/update_token/"
    data = {
        "email": settings.USER_EMAIL,
        "user_token": settings.USER_TOKEN,
    }
    response = requests.post(url, json=data)
    if (requests.status_codes.codes.get("ok") != response.status_code) or response.json().get('new_token'):
        raise agent_exceptions.FailedUpdateUserTokenException()
    return response.json().get('new_token')
