from rest_framework import serializers

from terra_ai.progress import pool as progress_pool


def progress_error(progress_name: str):
    def decorator(method):
        def wrapper(*args, **kwargs):
            progress = progress_pool(progress_name)
            print("DECORATOR:", progress)
            if not progress.success:
                progress.error.data = progress.data
                raise progress.error
            return method(*args, progress, **kwargs)

        return wrapper

    return decorator


def serialize_data(serializer: serializers.Serializer):
    def decorator(method):
        def wrapper(*args, **kwargs):
            _serializer = serializer(data=args[1].data)
            _serializer.is_valid(raise_exception=True)
            return method(*args, _serializer, **kwargs)

        return wrapper

    return decorator
