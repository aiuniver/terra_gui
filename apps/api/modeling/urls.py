from django.urls import path

from . import views


app_name = "modeling"

urlpatterns = [
    path("models/", views.ModelsAPIView.as_view(), name="models"),
    path("model/load/", views.ModelLoadAPIView.as_view(), name="model_load"),
    path(
        "model/load/progress/",
        views.ModelLoadProgressAPIView.as_view(),
        name="model_load_progress",
    ),
]
