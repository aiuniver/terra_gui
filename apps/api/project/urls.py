from django.urls import path

from . import views


app_name = "project"

urlpatterns = [
    path("name/", views.NameAPIView.as_view(), name="name"),
]
