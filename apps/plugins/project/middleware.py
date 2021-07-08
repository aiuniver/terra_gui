from django.utils.deprecation import MiddlewareMixin

from . import Project


class ProjectMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request.project = Project()
