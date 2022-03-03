from django.urls import path

from . import views


app_name = "version"

urlpatterns = [
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("", views.VersionAPIView.as_view(), name="version"),
]
