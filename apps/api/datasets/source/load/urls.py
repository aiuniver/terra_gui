from django.urls import path

from . import views


app_name = "load"

urlpatterns = [
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("", views.LoadAPIView.as_view(), name="load"),
]
