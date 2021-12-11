from enum import Enum
from typing import Optional
from ipaddress import IPv4Address
from pydantic import validator, BaseModel
from pydantic.types import PositiveInt

from rest_framework import serializers


class ServerStateValue(str, Enum):
    idle = "Ожидает настройки"
    waiting = "В процессе настройки"
    ready = "Готов к работе"
    error = "Ошибка настройки"


class ServerStateName(str, Enum):
    idle = "idle"
    waiting = "waiting"
    ready = "ready"
    error = "error"


class ServerStateData(BaseModel):
    name: ServerStateName = ServerStateName.idle
    value: Optional[ServerStateValue]
    error: Optional[str]

    @validator("value", always=True, pre=True)
    def _validate_value(cls, value, values):
        return ServerStateValue[values.get("name")]


class ServerData(BaseModel):
    id: PositiveInt
    state: ServerStateData
    domain_name: str
    ip_address: IPv4Address
    user: str
    port_ssh: PositiveInt = 22
    port_http: PositiveInt = 80
    port_https: PositiveInt = 443


class ServerFullData(ServerData):
    private_ssh_key: str
    public_ssh_key: str
    instruction: str


class CreateSerializer(serializers.Serializer):
    ip_address = serializers.IPAddressField()
    port_ssh = serializers.IntegerField(min_value=1, default=22)
    user = serializers.CharField()
    domain_name = serializers.CharField()
    port_http = serializers.IntegerField(min_value=1, default=80)
    port_https = serializers.IntegerField(min_value=1, default=443)


class ServerSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
