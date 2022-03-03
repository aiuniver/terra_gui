from django.urls import path

from . import views


app_name = "choice"

urlpatterns = [
    path("progress/", views.ProgressAPIView.as_view(), name="progress"),
    path("", views.ChoiceAPIView.as_view(), name="choice"),
]
