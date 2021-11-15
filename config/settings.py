"""
Django settings for cyber_kennel project.

Generated by 'django-admin startproject' using Django 3.1.7.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

import environ
import tempfile

from datetime import datetime


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = environ.Path(__file__) - 2


# Init environ
env = environ.Env()
env.read_env(BASE_DIR(".env"))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env.str("SECRET_KEY")


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env.bool("DEBUG", default=False)
USE_GPU = env.bool("USE_GPU", default=True)

ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=[])


# Application definition

DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
]
EXTERNAL_APPS = [
    "rest_framework",
]
INTERNAL_APPS = [
    "apps.api.apps.APIConfig",
    "apps.media.apps.MediaConfig",
    "apps.periodic_task.apps.PeriodicTaskConfig",
]

INSTALLED_APPS = DJANGO_APPS + EXTERNAL_APPS + INTERNAL_APPS

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.plugins.project.middleware.ProjectMiddleware",
]

SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [str(BASE_DIR("vue/dist"))],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = "ru"

TIME_ZONE = "Europe/Moscow"

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.1/howto/static-files/

VUE_ROOT = BASE_DIR("vue/dist")
VUE_URL = "/"


# Django Rest Framework

REST_FRAMEWORK = {
    "EXCEPTION_HANDLER": "apps.api.exceptions.handler",
}


# Terra AI

TERRA_AI_DATE_START = datetime.now()
TERRA_AI_BASE_DIR = env.str("TERRA_AI_BASE_DIR", default="/")
TERRA_AI_EXCHANGE_API_URL = env.str("TERRA_AI_EXCHANGE_API_URL")
TERRA_AI_DATA_PATH = env.str("TERRA_AI_DATA_PATH")
TERRA_AI_PROJECT_PATH = f"{tempfile.gettempdir()}/tai-project"


# User data

USER_PORT = env.int("USER_PORT", default=9120)
USER_LOGIN = env.str("USER_LOGIN")
USER_NAME = env.str("USER_NAME")
USER_LASTNAME = env.str("USER_LASTNAME")
USER_EMAIL = env.str("USER_EMAIL")
USER_TOKEN = env.str("USER_TOKEN")
USER_SERVERS = env.json("USER_SERVERS", default=[])
