from django.urls import path, include

from . import views


app_name = "datasets"

urlpatterns = [
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("versions/", views.VersionsAPIView.as_view(), name="versions"),
    path("delete/", include("apps.api.datasets.delete.urls", namespace="delete")),
    path("choice/", include("apps.api.datasets.choice.urls", namespace="choice")),
    path("create/", include("apps.api.datasets.create.urls", namespace="create")),
    path("source/", include("apps.api.datasets.source.urls", namespace="source")),
]
