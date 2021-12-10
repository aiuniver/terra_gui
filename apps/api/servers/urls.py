from django.urls import path

from . import views


app_name = "servers"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("list/", views.ListAPIView.as_view(), name="list"),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path("setup/", views.SetupAPIView.as_view(), name="setup"),
    path("ready/", views.ReadyAPIView.as_view(), name="ready"),
]
