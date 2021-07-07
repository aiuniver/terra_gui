from django.urls import path

from . import views


app_name = "datasets"

urlpatterns = [
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("sources/", views.SourcesAPIView.as_view(), name="sources"),
]
