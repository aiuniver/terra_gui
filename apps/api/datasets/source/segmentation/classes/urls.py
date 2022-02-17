from django.urls import path

from . import views


app_name = "classes"

urlpatterns = [
    path("autosearch/", views.AutosearchAPIView.as_view(), name="autosearch"),
    path("annotation/", views.AnnotationAPIView.as_view(), name="annotation"),
]
