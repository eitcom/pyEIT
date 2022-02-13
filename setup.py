# copied from https://github.com/adrienverge/yamllint/
import pathlib
from setuptools import setup

from pyeit import (
    __author__,
    __author_email__,
    __license__,
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
)


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
    name=APP_NAME,
    version=APP_VERSION,
    author=__author__,
    author_email=__author_email__,
    description=APP_DESCRIPTION.split("\n")[0],
    long_description=README,
    long_description_content_type="text/markdown",
    license=__license__,
)
