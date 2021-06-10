import json
import base64

from django.http import HttpResponse
from django.views.generic import View, TemplateView

from apps.plugins.terra import terra_exchange
from apps.plugins.terra.neural import colab_exchange


class ProjectViewMixin(TemplateView):
    pass


class ConfigJSView(View):
    def get(self, request, *arg, **kwargs):
        data = base64.b64encode(
            json.dumps(terra_exchange.project.dict()).encode("utf-8")
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

    def dispatch(self, request, *args, **kwargs):
        colab_exchange.out_data.update(terra_exchange.project.dir.training_output)
        return super().dispatch(request, *args, **kwargs)
