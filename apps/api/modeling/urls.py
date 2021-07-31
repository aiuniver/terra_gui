from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
]
