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

import mimetypes
import posixpath

from pathlib import Path

from django.urls import path
from django.http import HttpResponseNotModified, HttpResponseNotFound, FileResponse
from django.conf import settings
from django.conf.urls import include
from django.conf.urls.static import static
from django.utils._os import safe_join
from django.utils.http import http_date
from django.views.static import was_modified_since


def static_view(request, path, document_root=None):
    path = posixpath.normpath(path).lstrip("/")
    fullpath = Path(safe_join(document_root, path))
    if fullpath.is_dir():
        return HttpResponseNotFound()
    if not fullpath.exists():
        return HttpResponseNotFound()
    statobj = fullpath.stat()
    if not was_modified_since(
        request.META.get("HTTP_IF_MODIFIED_SINCE"), statobj.st_mtime, statobj.st_size
    ):
        return HttpResponseNotModified()
    content_type, encoding = mimetypes.guess_type(str(fullpath))
    content_type = content_type or "application/octet-stream"
    response = FileResponse(fullpath.open("rb"), content_type=content_type)
    response.headers["Last-Modified"] = http_date(statobj.st_mtime)
    if encoding:
        response.headers["Content-Encoding"] = encoding
    return response


urlpatterns = (
    [
        path("api/v1/", include("apps.api.urls", namespace="apps_api")),
        path("_media/", include("apps.media.urls", namespace="apps_media")),
    ]
    + static(settings.STATIC_URL, view=static_view, document_root=settings.STATIC_ROOT)
    + [
        path("", include("apps.core.urls", namespace="apps_core")),
    ]
)
