from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("get/", views.GetAPIView.as_view(), name="get"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("clear/", views.ClearAPIView.as_view(), name="clear"),
    path("update/", views.UpdateAPIView.as_view(), name="update"),
    path("validate/", views.ValidateAPIView.as_view(), name="validate"),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path("preview/", views.PreviewAPIView.as_view(), name="preview"),
    path("delete/", views.DeleteAPIView.as_view(), name="delete"),
    path("datatype/", views.DatatypeAPIView.as_view(), name="datatype"),
]
