import os
import re
from setuptools import setup

def get_version():
    with open(os.path.join("pyTomoAO", "__init__.py")) as f:
        return re.search(r'__version__ = "(.*)"', f.read()).group(1)
setup(
    name='pyTomoAO',
    version=get_version(),
    description='An open-source tool for tomographic reconstuction for AO systems',
    url='https://github.com/KeckObservatory/pyTomoAO',
    author='Jacob Taylor',
    author_email='jtaylor@keck.hawaii.edu',
    license='MIT',
    packages=['pyTomoAO'],
    install_requires=[
        'numpy',
        'matplotlib',
        'numba',
        'scipy',
        'pytest',
        'PyYAML'
    ],
    project_urls={
        'Documentation': 'https://keckobservatory.github.io/pyTomoAO/',
        'Source': 'https://github.com/KeckObservatory/pyTomoAO',
        'Issues': 'https://github.com/KeckObservatory/pyTomoAO/issues',
    },
    extras_require={
        # Keep in sync with docs/requirements.txt.
        'docs': [
            'sphinx>=7.2,<9',
            'furo>=2024.1.29',
            'myst-parser>=2.0',
            'sphinx-autodoc-typehints>=2.0',
            'sphinx-copybutton>=0.5.2',
            'sphinx-design>=0.5',
            'sphinxcontrib-mermaid>=0.9',
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Environment :: MacOS X',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
