from django.utils.deprecation import MiddlewareMixin

from . import terra_exchange


def collect_filters_datasets(datasets: dict, tags: dict) -> dict:
    def parse_dataset_filters(filters: list) -> list:
        items = list(filter(None, filters))
        return dict(
            map(
                lambda item: [list(tags.keys())[list(tags.values()).index(item)], item],
                items,
            )
        )

    output = {}
    for name, items in datasets.items():
        output[name] = {
            "tasks": list(
                map(
                    lambda item: list(tags.keys())[list(tags.values()).index(item)],
                    items[0][1:],
                )
            ),
            "filters": parse_dataset_filters(items[0] + datasets.get(name, [])[1:]),
        }

    return output


class TerraProjectMiddleware(MiddlewareMixin):
    def process_request(self, request):
        response = terra_exchange.call("get_state", task=terra_exchange.project.task)
        if response.success:
            tags = response.data.get("tags", {})
            datasets = collect_filters_datasets(response.data.get("datasets", {}), tags)
            terra_exchange.project.datasets = datasets
            terra_exchange.project.tags = tags
            terra_exchange.project.layers_types = response.data.get("layers_types", {})
            terra_exchange.project.optimizers = response.data.get("optimizers", [])
            terra_exchange.project.compile = response.data.get("compile", {})
            terra_exchange.project.callbacks = (
                terra_exchange.project.callbacks
                if len(terra_exchange.project.callbacks.keys())
                else response.data.get("callbacks", {})
            )
            terra_exchange.project.hardware = response.data.get("hardware", "CPU")
            terra_exchange.project.error = ""
        else:
            terra_exchange.project = {"error": "No connection to TerraAI project"}
        request.terra_project = terra_exchange.project.__dict__
