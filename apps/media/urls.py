from django.urls import path

from . import views


app_name = "apps_media"

urlpatterns = [
    path("blank/", views.BlankView.as_view(), name="blank"),
]
