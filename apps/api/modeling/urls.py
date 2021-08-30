from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("clear/", views.ClearAPIView.as_view(), name="clear"),
    path("update/", views.UpdateAPIView.as_view(), name="update"),
    path("layer/save/", views.LayerSaveAPIView.as_view(), name="layer_save"),
    path("validate/", views.ValidateAPIView.as_view(), name="validate"),
]
