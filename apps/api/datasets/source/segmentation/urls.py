from django.urls import path, include


app_name = "segmentation"

urlpatterns = [
    path(
        "classes/",
        include(
            "apps.api.datasets.source.segmentation.classes.urls", namespace="classes"
        ),
    ),
]
