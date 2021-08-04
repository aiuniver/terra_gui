from django.urls import path

from . import views


app_name = "deploy"

urlpatterns = [
    path("upload/", views.UploadAPIView.as_view(), name="upload"),
]
