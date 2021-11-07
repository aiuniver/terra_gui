from enum import Enum

import dotenv

from django.conf import settings

env_file = settings.BASE_DIR(".env")
dotenv.load_dotenv(env_file)


class Keys(str, Enum):
    login = 'USER_LOGIN'
    first_name = 'USER_NAME'
    last_name = 'USER_LASTNAME'
    email = 'USER_EMAIL'
    token = 'USER_TOKEN'


def update_env_file(**kwargs):
    for key, value in kwargs.items():
        key = getattr(Keys, key).value
        dotenv.set_key(env_file, key, value)
