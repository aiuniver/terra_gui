from django.urls import path

from . import views as project_views


app_name = "apps_project"

urlpatterns = [
    path("config.js", project_views.ConfigJSView.as_view(), name="config_js"),
    path("datasets/", project_views.DatasetsView.as_view(), name="datasets"),
    path("modeling/", project_views.ModelingView.as_view(), name="modeling"),
    path("training/", project_views.TrainingView.as_view(), name="training"),
]
