from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("update/", views.UpdateAPIView.as_view(), name="update"),
    path("layer/save/", views.LayerSaveAPIView.as_view(), name="layer_save"),
]
