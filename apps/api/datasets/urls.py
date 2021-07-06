from django.urls import path

from . import views


app_name = "datasets"

urlpatterns = [
    path("sources/", views.SourcesAPIView.as_view(), name="sources"),
]
