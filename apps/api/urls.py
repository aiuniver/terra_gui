from django.urls import path

from . import views as api_views


app_name = "apps_api"

urlpatterns = [
    path("exchange/<name>/", api_views.ExchangeAPIView.as_view(), name="exchange"),
    path("layers-types/", api_views.LayersTypesAPIView.as_view(), name="layers-types"),
]
