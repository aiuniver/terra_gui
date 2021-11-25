from rest_framework import serializers


class CreateSerializer(serializers.Serializer):
    ip_address = serializers.IPAddressField()
    port_ssh = serializers.IntegerField(min_value=1, default=22)
    user = serializers.CharField()
    password = serializers.CharField()
    domain_name = serializers.CharField()
    port_http = serializers.IntegerField(min_value=1, default=80)
    port_https = serializers.IntegerField(min_value=1, default=443)