from django.urls import path, re_path
from django.conf.urls import include

from . import views


app_name = "apps_api"

urlpatterns = [
    path("config/", views.ConfigAPIView.as_view(), name="config"),
    path("common/", include("apps.api.common.urls", namespace="common")),
    path("profile/", include("apps.api.profile.urls", namespace="profile")),
    path("project/", include("apps.api.project.urls", namespace="project")),
    path("datasets/", include("apps.api.datasets.urls", namespace="datasets")),
    path("modeling/", include("apps.api.modeling.urls", namespace="modeling")),
    path("training/", include("apps.api.training.urls", namespace="training")),
    path("cascades/", include("apps.api.cascades.urls", namespace="cascades")),
    path("deploy/", include("apps.api.deploy.urls", namespace="deploy")),
    re_path("^.*", views.NotFoundAPIView.as_view(), name="not_found"),
]
