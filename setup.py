from distutils.core import setup
from setuptools import find_packages


setup(
    name="FSM_ard-hard",
    python_requires=">=3.8",
    packages=find_packages(include=["ard-hard*"]),
    install_requires=[
        "matplotlib",
        "pandas",
        "scikit-learn",
    ],
)
