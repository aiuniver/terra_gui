from django.urls import path

from . import views


app_name = "create"

urlpatterns = [
    path("version/", views.VersionAPIView.as_view(), name="version"),
    path("validate/", views.ValidateAPIView.as_view(), name="validate"),
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("", views.CreateAPIView.as_view(), name="create"),
]
