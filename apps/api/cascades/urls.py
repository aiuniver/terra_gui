from django.urls import path

from . import views


app_name = "cascades"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("update/", views.UpdateAPIView.as_view(), name="update"),
]
