from django.urls import path

from . import views


app_name = "common"

urlpatterns = [
    path("logs/", views.LogsAPIView.as_view(), name="logs"),
    path(
        "validate-dataset-model/",
        views.ValidateDatasetModelAPIView.as_view(),
        name="validate_dataset_model",
    ),
]
