import json
import base64

from django.http import HttpResponse
from django.views.generic import View, TemplateView


class ProjectViewMixin(TemplateView):
    pass


class ConfigJSView(View):
    def get(self, request, *arg, **kwargs):
        data = base64.b64encode(
            json.dumps(request.terra_project).encode("utf-8")
        ).decode("utf-8")
        return HttpResponse(
            f'window._terra_project="{data}"',
            content_type="application/javascript",
        )


class DatasetsView(ProjectViewMixin):
    template_name = "project/datasets.html"


class ModelingView(ProjectViewMixin):
    template_name = "project/modeling.html"


class TrainingView(ProjectViewMixin):
    template_name = "project/training.html"

    def get_context_data(self, **kwargs):
        kwargs = super().get_context_data(**kwargs)
        # kwargs["optimizers"] = settings.TERRA_EXCHANGE.get_optimizers_list()
        # try:
        #     kwargs["callbacks"] = settings.TERRA_EXCHANGE.get_callbacks_switches()
        # except KeyError:
        #     kwargs["callbacks"] = {}
        return kwargs
