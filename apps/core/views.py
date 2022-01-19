from django.conf import settings
from django.views.generic import TemplateView


class MainView(TemplateView):
    template_name = "index.html"

    def get_template_names(self):
        if not settings.USER_SESSION:
            self.template_name = "login.html"
        return super().get_template_names()
