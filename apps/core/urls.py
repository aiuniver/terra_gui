from django.urls import re_path

from . import views


app_name = "apps_core"

urlpatterns = [
    re_path("^.*", views.MainView.as_view(), name="main"),
]
