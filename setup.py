import os
import re

from setuptools import setup


def get_version():
    with open(os.path.join("pyTomoAO", "__init__.py")) as f:
        return re.search(r'__version__ = "(.*)"', f.read()).group(1)


def get_long_description():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


setup(
    name="pyTomoAO",
    version=get_version(),
    description="An open-source tool for tomographic reconstuction for AO systems",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/KeckObservatory/pyTomoAO",
    author="Jacob Taylor",
    author_email="jacobataylor7@gmail.com",
    license="MIT",
    packages=["pyTomoAO"],
    # The reference YAML configurations ship inside the package so that `pip install
    # pyTomoAO` is enough to run the documented examples; see pyTomoAO.example_config.
    package_data={"pyTomoAO": ["data/*.yaml"]},
    include_package_data=True,
    python_requires=">=3.9",
    # Runtime dependencies only. Test and documentation tooling lives in the
    # extras below so that installing pyTomoAO does not drag them in.
    install_requires=["numpy", "matplotlib", "numba", "scipy", "PyYAML"],
    project_urls={
        "Documentation": "https://keckobservatory.github.io/pyTomoAO/",
        "Source": "https://github.com/KeckObservatory/pyTomoAO",
        "Issues": "https://github.com/KeckObservatory/pyTomoAO/issues",
    },
    extras_require={
        # Keep in sync with docs/requirements.txt.
        "docs": [
            "sphinx>=7.2,<9",
            "furo>=2024.1.29",
            "myst-parser>=2.0",
            "sphinx-autodoc-typehints>=2.0",
            "sphinx-copybutton>=0.5.2",
            "sphinx-design>=0.5",
            "sphinxcontrib-mermaid>=0.9",
        ],
        # Keep the ruff pin in sync with .github/workflows/code-health.yml.
        "dev": [
            "ruff==0.15.5",
            "pytest",
            "pytest-cov",
        ],
        # CUDA 12 build of CuPy. Users on CUDA 11 should install cupy-cuda11x instead;
        # pyTomoAO detects whichever is importable at import time.
        "gpu": ["cupy-cuda12x"],
    },
    classifiers=[
        # Keep in sync with python_requires and the CI matrix in
        # .github/workflows/test.yml.
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Environment :: MacOS X",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
