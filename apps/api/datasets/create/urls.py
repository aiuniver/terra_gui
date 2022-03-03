from django.urls import path, include

from . import views


app_name = "create"

urlpatterns = [
    path("validate/", views.ValidateAPIView.as_view(), name="validate"),
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("", views.CreateAPIView.as_view(), name="create"),
    path(
        "version/",
        include("apps.api.datasets.create.version.urls", namespace="version"),
    ),
]
