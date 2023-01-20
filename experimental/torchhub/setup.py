from setuptools import find_namespace_packages, setup

setup(
    name="liga-torchhub",
    version="0.1.0",
    license="Apache License, Version 2.0",
    author="Rikai authors",
    author_email="liga-dev@eto.ai",
    url="https://github.com/eto-ai/liga",
    python_requires=">=3.7",
    install_requires=["liga>=0.1.1", "torch", "torchvision"],
    extras_require={
        "dev": [
            "black",
            "isort",
            # for testing
            "pytest",
        ]
    },
    packages=find_namespace_packages(include=["liga.*"]),
)
