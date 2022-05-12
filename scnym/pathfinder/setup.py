import sys
if sys.version_info < (3,):
    sys.exit('pathfinder requires Python >= 3.6')
from pathlib import Path

from setuptools import setup, find_packages

__author__ = "Jacob C. Kimmel"
__email__ = "jacobkimmel@gmail.com"

setup(
    name='pathfinder',
    version="0.0.1",
    description="Rationale design of reprogramming strategies",
    long_description=" ",
    url=' ',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
