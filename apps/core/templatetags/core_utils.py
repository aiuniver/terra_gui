from django import template
from django.urls import resolve

from apps.plugins.terra import terra_exchange
from apps.plugins.terra.data import OutputConfig


register = template.Library()


@register.filter
def menu_active(value: str, request) -> str:
    resolver = resolve(request.path)
    return " active" if f"{resolver.namespace}:{resolver.url_name}" == value else ""


@register.filter
def dataset_tags_string(tags: list) -> str:
    items = list(map(lambda item: f"filter-{item}", tags))
    return " ".join(items)


@register.filter
def apps_namespace(request) -> str:
    resolver = resolve(request.path)
    return resolver.namespace


@register.filter
def apps_namespace_page(request) -> str:
    resolver = resolve(request.path)
    return resolver.url_name


@register.simple_tag
def task_loss_selected(output: str, task: str, loss: str) -> bool:
    config: OutputConfig = terra_exchange.project.training.outputs.get(output)
    if not config:
        return ""
    return (
        ' selected="selected"'
        if f"{config.task.value}_{config.loss}" == f"{task}_{loss}"
        else ""
    )
