from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("models/", views.ModelsAPIView.as_view(), name="models"),
]
