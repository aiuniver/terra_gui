from django import template
from django.urls import resolve


register = template.Library()


@register.filter
def menu_active(value: str, request) -> str:
    resolver = resolve(request.path)
    return " active" if f"{resolver.namespace}:{resolver.url_name}" == value else ""


@register.filter
def dataset_filters_string(filters: list) -> str:
    items = list(map(lambda item: f"filter-{item}", filters))
    return " ".join(items)


@register.filter
def apps_namespace(request) -> str:
    resolver = resolve(request.path)
    return resolver.namespace


@register.filter
def apps_namespace_page(request) -> str:
    resolver = resolve(request.path)
    return resolver.url_name
