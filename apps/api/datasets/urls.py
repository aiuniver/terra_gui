from django.urls import path

from . import views


app_name = "datasets"

urlpatterns = [
    path("choice/", views.ChoiceAPIView.as_view(), name="choice"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path(
        "source/load/progress/",
        views.SourceLoadProgressAPIView.as_view(),
        name="source_load_progress",
    ),
    path("source/load/", views.SourceLoadAPIView.as_view(), name="source_load"),
    path("source/create/", views.SourcesCreateAPIView.as_view(), name="source_create"),
    path("sources/", views.SourcesAPIView.as_view(), name="sources"),
]
