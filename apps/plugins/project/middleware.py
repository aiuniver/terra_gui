from django.utils.deprecation import MiddlewareMixin

from . import project


class ProjectMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.project = project
