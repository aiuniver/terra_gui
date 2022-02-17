from django.urls import path, include

from . import views


app_name = "datasets"

urlpatterns = [
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("versions/", views.VersionsAPIView.as_view(), name="versions"),
    path("delete/", views.DeleteAPIView.as_view(), name="delete"),
    path("choice/", include("apps.api.datasets.choice.urls", namespace="choice")),
    path("create/", include("apps.api.datasets.create.urls", namespace="create")),
    path("source/", include("apps.api.datasets.source.urls", namespace="source")),
]
