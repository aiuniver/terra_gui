from django.urls import path

from . import views


app_name = "project"

urlpatterns = [
    path("name/", views.NameAPIView.as_view(), name="name"),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path("save/", views.SaveAPIView.as_view(), name="save"),
    path("load/", views.LoadAPIView.as_view(), name="load"),
]
