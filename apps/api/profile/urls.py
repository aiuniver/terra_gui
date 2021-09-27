from django.urls import path

from . import views


app_name = "profile"

urlpatterns = [
    path("save/", views.SaveAPIView.as_view(), name="save"),
]
