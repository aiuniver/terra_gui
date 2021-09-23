from django.apps import apps


def path_hash(**kwargs):
    return apps.get_app_config("media").path_hash(**kwargs)
