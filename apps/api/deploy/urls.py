from django.urls import path

from . import views


app_name = "deploy"

urlpatterns = [
    path("prepare/", views.PrepareAPIView.as_view(), name="prepare"),
    path("upload/", views.UploadAPIView.as_view(), name="upload"),
]
