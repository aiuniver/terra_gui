from django.urls import path

from . import views as api_views


app_name = "apps_api"

urlpatterns = [
    path("exchange/<name>/", api_views.ExchangeAPIView.as_view(), name="exchange"),
]
