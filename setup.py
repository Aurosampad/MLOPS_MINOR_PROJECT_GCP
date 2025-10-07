from setuptools import setup,find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='MLOPS_MINOR_PROJECT_GCP',
    version='0.1',
    author='Aurosampad',
    packages=find_packages(),
    install_requires = required,
)