from rest_framework import serializers


class CreateSerializer(serializers.Serializer):
    ip_address = serializers.IPAddressField()
    port_ssh = serializers.IntegerField(min_value=1, default=22)
    user = serializers.CharField()
    domain_name = serializers.CharField()
    port_http = serializers.IntegerField(min_value=1, default=80)
    port_https = serializers.IntegerField(min_value=1, default=443)


class InstructionSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)


class SetupSerializer(serializers.Serializer):
    id = serializers.IntegerField(min_value=1)
