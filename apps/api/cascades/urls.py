from django.urls import path

from . import views


app_name = "cascades"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("update/", views.UpdateAPIView.as_view(), name="update"),
    path("clear/", views.ClearAPIView.as_view(), name="clear"),
    path("validate/", views.ValidateAPIView.as_view(), name="validate"),
    path("start/", views.StartAPIView.as_view(), name="start"),
    path("save/", views.SaveAPIView.as_view(), name="save"),
    path("preview/", views.PreviewAPIView.as_view(), name="preview"),
]
