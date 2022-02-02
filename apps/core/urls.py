from django.urls import path, re_path

from . import views


app_name = "apps_core"

urlpatterns = [
    path("notebook/", views.NotebookView.as_view(), name="notebook"),
    re_path("^.*", views.MainView.as_view(), name="main"),
]
