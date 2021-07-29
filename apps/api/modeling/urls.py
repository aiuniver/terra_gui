from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("model/load/", views.ModelLoadAPIView.as_view(), name="model_load"),
    path("models/", views.ModelsAPIView.as_view(), name="models"),
]
