import json

from django.conf import settings

from apps.plugins.frontend import defaults_data

from . import base


class NotFoundAPIView(base.BaseAPIView):
    pass


class TestAPIView(base.BaseAPIView):
    def post(self, request, **kwargs):
        return base.BaseResponseError(
            {
                "level": "error",
                "datetime": 1638776928,
                "title": "error() missing 1 required positional argument: 'msg'",
                "message": """Traceback (most recent call last):
  File "/usr/lib/python3.7/threading.py", line 926, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.7/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/utils/autoreload.py", line 64, in wrapper
    fn(*args, **kwargs)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/core/management/commands/runserver.py", line 118, in inner_run
    self.check(display_num_errors=True)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/core/management/base.py", line 423, in check
    databases=databases,
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/core/checks/registry.py", line 76, in run_checks
    new_errors = check(app_configs=app_configs, databases=databases)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/core/checks/urls.py", line 13, in check_url_config
    return check_resolver(resolver)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/core/checks/urls.py", line 23, in check_resolver
    return check_method()
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/urls/resolvers.py", line 412, in check
    for pattern in self.url_patterns:
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/utils/functional.py", line 48, in __get__
    res = instance.__dict__[self.name] = self.func(instance)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/urls/resolvers.py", line 598, in url_patterns
    patterns = getattr(self.urlconf_module, "urlpatterns", self.urlconf_module)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/utils/functional.py", line 48, in __get__
    res = instance.__dict__[self.name] = self.func(instance)
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/urls/resolvers.py", line 591, in urlconf_module
    return import_module(self.urlconf_name)
  File "/usr/lib/python3.7/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/config/urls.py", line 54, in <module>
    path("api/v1/", include("apps.api.urls", namespace="apps_api")),
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/urls/conf.py", line 34, in include
    urlconf_module = import_module(urlconf_module)
  File "/usr/lib/python3.7/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/apps/api/urls.py", line 11, in <module>
    path("common/", include("apps.api.common.urls", namespace="common")),
  File "/home/bl146u/Virtualenvs/terra_gui/lib/python3.7/site-packages/django/urls/conf.py", line 34, in include
    urlconf_module = import_module(urlconf_module)
  File "/usr/lib/python3.7/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
  File "<frozen importlib._bootstrap>", line 983, in _find_and_load
  File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/apps/api/common/urls.py", line 3, in <module>
    from . import views
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/apps/api/common/views.py", line 2, in <module>
    from terra_ai.agent import agent_exchange
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/terra_ai/agent/__init__.py", line 47, in <module>
    from ..logging import logger
  File "/home/bl146u/Virtualenvs/terra_gui/terra_gui/terra_ai/logging.py", line 4, in <module>
    logger.error()
TypeError: error() missing 1 required positional argument: 'msg'""",
            }
        )


class ConfigAPIView(base.BaseAPIView):
    def post(self, request, **kwargs):
        return base.BaseResponseSuccess(
            {
                "defaults": json.loads(defaults_data.json()),
                "project": json.loads(request.project.frontend()),
                "user": {
                    "login": settings.USER_LOGIN,
                    "first_name": settings.USER_NAME,
                    "last_name": settings.USER_LASTNAME,
                    "email": settings.USER_EMAIL,
                    "token": settings.USER_TOKEN,
                },
            }
        )
