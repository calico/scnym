import sys
if sys.version_info < (3,):
    sys.exit('scnym requires Python >= 3.6')
from pathlib import Path

from setuptools import setup, find_packages

try:
    from scnym import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(
    name='scnym',
    version='0.1',
    description='Neural network classifiers for single cell profiling',
    long_description=Path('README.md').read_text('utf-8'),
    url='http://github.com/calico/scnym',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    packages=find_packages(),
    entry_points=dict(
        console_scripts=['scnym=scnym.main:main'],
    ),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)