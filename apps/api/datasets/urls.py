from django.urls import path

from . import views


app_name = "datasets"

urlpatterns = [
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("versions/", views.VersionsAPIView.as_view(), name="versions"),
    path(
        "choice/progress/",
        views.ChoiceProgressAPIView.as_view(),
        name="choice_progress",
    ),
    path("choice/", views.ChoiceAPIView.as_view(), name="choice"),
    path(
        "create/progress/",
        views.CreateProgressAPIView.as_view(),
        name="create_progress",
    ),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path(
        "source/load/progress/",
        views.SourceLoadProgressAPIView.as_view(),
        name="source_load_progress",
    ),
    path("source/load/", views.SourceLoadAPIView.as_view(), name="source_load"),
    path(
        "source/segmentation/classes/autosearch/",
        views.SourceSegmentationClassesAutoSearchAPIView.as_view(),
        name="source_segmentation_classes_autosearch",
    ),
    path(
        "source/segmentation/classes/annotation/",
        views.SourceSegmentationClassesAnnotationAPIView.as_view(),
        name="source_segmentation_classes_annotation",
    ),
    path("sources/", views.SourcesAPIView.as_view(), name="sources"),
    path("delete/", views.DeleteAPIView.as_view(), name="delete"),
]
