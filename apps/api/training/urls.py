from django.urls import path

from . import views


app_name = "training"

urlpatterns = [
    path("start/", views.StartAPIView.as_view(), name="start"),
    path("stop/", views.StopAPIView.as_view(), name="stop"),
    path("clear/", views.ClearAPIView.as_view(), name="clear"),
    path("interactive/", views.InteractiveAPIView.as_view(), name="interactive"),
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("save/", views.SaveAPIView.as_view(), name="save"),
    path("update/", views.UpdateAPIView.as_view(), name="change"),
]
