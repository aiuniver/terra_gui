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
        response = terra_exchange.call("get_state")
        if response.success:
            tags = response.data.get("tags", {})
            datasets = collect_filters_datasets(response.data.get("datasets", {}), tags)
            terra_exchange.project.datasets = datasets
            terra_exchange.project.tags = tags
        request.terra_project = terra_exchange.project.__dict__
