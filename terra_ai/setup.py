#!/usr/bin/env python3

from setuptools import setup, find_packages
from . import __version__


setup(
    name="terra_ai",
    version=__version__,
    packages=find_packages(),
)
