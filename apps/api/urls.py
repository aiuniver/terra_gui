from django.urls import path
from django.conf.urls import include


app_name = "apps_api"

urlpatterns = [
    path("datasets/", include("apps.api.datasets.urls", namespace="datasets")),
    path("modeling/", include("apps.api.modeling.urls", namespace="modeling")),
    path("training/", include("apps.api.training.urls", namespace="training")),
]
