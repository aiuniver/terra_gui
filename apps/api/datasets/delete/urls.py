from django.urls import path

from . import views


app_name = "delete"

urlpatterns = [
    path("version/", views.DeleteVersionAPIView.as_view(), name="version"),
    path("", views.DeleteAPIView.as_view(), name="delete"),
]
