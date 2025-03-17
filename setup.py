from setuptools import setup, find_packages

setup(
    name="simple_eval_harness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "math-verify",
        "datasets",
        "sqlitedict",
        "tenacity",
    ],
    author="eyuansu71",
    author_email="eyuansu71@gmail.com",
    description="A simple evaluation harness",
)