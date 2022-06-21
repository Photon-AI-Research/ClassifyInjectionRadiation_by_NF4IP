
from setuptools import setup, find_packages
from lwfa.core.version import get_version

VERSION = get_version()

f = open('README.md', 'r')
LONG_DESCRIPTION = f.read()
f.close()

setup(
    name='lwfa',
    version=VERSION,
    description='LWFA',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='John Doe',
    author_email='john.doe@example.com',
    url='https://github.com/johndoe/myapp/',
    license='unlicensed',
    packages=find_packages(exclude=['ez_setup', 'tests*']),
    package_data={'lwfa': ['templates/*']},
    include_package_data=True,
    entry_points="""
        [console_scripts]
        lwfa = lwfa.main:main
    """,
)
