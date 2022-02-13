import pathlib
from setuptools import setup, find_packages


VERSION = "1.1.2"
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="pyeit",
    version=VERSION,
    description="A Python based framework for EIT (Electrical Impedance Tomography)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/liubenyuan/pyEIT",
    author="Benyuan Liu",
    author_email="liubenyuan@gmail.com",
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["Electrical impedance tomography", "Python", "Unstructural Mesh"],
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
    install_requires=["numpy", "scipy", "pandas"],
)
