from setuptools import setup

setup(
    name='pyTomoAO',
    version='1.0.0',    
    description='An open-source tool for tomographic reconstuction for AO systems',
    url='https://github.com/jacotay7/pyTomoAO',
    author='Jacob Taylor',
    author_email='jtaylor@keck.hawaii.edu',
    license='GNU',
    packages=['pyTomoAO'],
    install_requires=[
        'numpy',
        'matplotlib',
        'numba',
        'scipy',
        'pytest'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'sphinx-autodoc-typehints'
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
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
