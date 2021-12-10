from django.urls import path

from . import views


app_name = "servers"

urlpatterns = [
    path("list/", views.ListAPIView.as_view(), name="list"),
    path("create/", views.CreateAPIView.as_view(), name="create"),
    path("instruction/", views.InstructionAPIView.as_view(), name="instruction"),
    path("setup/", views.SetupAPIView.as_view(), name="setup"),
]
