from pathlib import Path

from setuptools import setup, find_packages

try:
    from scnym import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

# Single source of truth for version
_version = Path('VERSION').read_text('utf-8').strip()

setup(
    name='scnym',
    version=_version,
    description="Semi supervised adversarial neural networks for single cell classification",
    long_description="scNym uses the semi-supervised MixMatch framework and domain adversarial training to take advantage of information in both the labeled and unlabeled datasets.",
    url='http://github.com/calico/scnym',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.10',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
        if l.strip()
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'ruff',
        ],
    },
    packages=find_packages(),
    entry_points=dict(
        console_scripts=['scnym=scnym.main:main', 'scnym_ad=scnym.scnym_ad:main'],
    ),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
