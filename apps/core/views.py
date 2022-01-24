from django.conf import settings
from django.views.generic import TemplateView
from django.http.response import HttpResponseRedirect

from apps.api import remote


class MainView(TemplateView):
    template_name = "index.html"

    def get_template_names(self):
        if not settings.USER_SESSION:
            self.template_name = "login.html"
        return super().get_template_names()

    def dispatch(self, request, *args, **kwargs):
        if settings.USER_SESSION:
            settings.USER = remote.request("/user/")
            if not settings.USER:
                settings.USER_SESSION = None
                return HttpResponseRedirect("/")
        return super().dispatch(request, *args, **kwargs)
