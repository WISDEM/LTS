from setuptools import setup

from os import path
from io import open


metadata = dict(
    name="lts",
    version="0.0.1",
    description="Design model for low-temperature superconducting generators",
    author="Latha Sethuraman",
    packages=["lts"],
    python_requires=">=3.6",
    zip_safe=True,
    )

setup(**metadata)