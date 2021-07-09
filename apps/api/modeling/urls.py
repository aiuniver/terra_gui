from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("model/", views.ModelAPIView.as_view(), name="model"),
    path("models/", views.ModelsAPIView.as_view(), name="models"),
]
