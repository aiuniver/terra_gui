from django.urls import path

from . import views


app_name = "project"

urlpatterns = [
    path("name/", views.NameAPIView.as_view(), name="name"),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path("save/", views.SaveAPIView.as_view(), name="save"),
    path("info/", views.InfoAPIView.as_view(), name="info"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
    path("load/progress/", views.LoadProgressAPIView.as_view(), name="load_progress"),
    path("delete/", views.DeleteAPIView.as_view(), name="delete"),
]
