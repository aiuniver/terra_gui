"""cyber_kennel URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from django.conf import settings
from django.conf.urls import include
from django.conf.urls.static import static

from apps.core import views as core_views


handler404 = core_views.handler404


urlpatterns = [
    path("api/v1/", include("apps.api.urls", namespace="apps_api")),
    path("project/", include("apps.project.urls", namespace="apps_project")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
