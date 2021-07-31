from django.urls import path

from . import views


app_name = "deploy"

urlpatterns = [
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("upload/", views.UploadAPIView.as_view(), name="upload"),
]
