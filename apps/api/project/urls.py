from django.urls import path

from . import views


app_name = "project"

urlpatterns = [
    path("config/", views.ConfigAPIView.as_view(), name="config"),
    path("defaults/", views.DefaultsAPIView.as_view(), name="defaults"),
]
