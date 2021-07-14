from django.urls import path

from . import views


app_name = "datasets"

urlpatterns = [
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("source/load/", views.SourceLoadAPIView.as_view(), name="source_load"),
    path(
        "source/load/progress/",
        views.SourceLoadProgressAPIView.as_view(),
        name="source_load_progress",
    ),
    path("sources/", views.SourcesAPIView.as_view(), name="sources"),
]
