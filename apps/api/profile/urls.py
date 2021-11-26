from django.urls import path
from django.conf.urls import include

from . import views


app_name = "profile"

urlpatterns = [
    path("save/", views.SaveAPIView.as_view(), name="save"),
    path("update_token/", views.UpdateTokenAPIView.as_view(), name="update_token"),
    path("servers/", include("apps.api.profile.servers.urls", namespace="servers")),
]
