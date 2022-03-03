from django.urls import path, include

from . import views


app_name = "source"

urlpatterns = [
    path("load/", include("apps.api.datasets.source.load.urls", namespace="load")),
    path(
        "segmentation/",
        include("apps.api.datasets.source.segmentation.urls", namespace="segmentation"),
    ),
    path("", views.SourceAPIView.as_view(), name="source"),
]
