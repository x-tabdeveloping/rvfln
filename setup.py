import setuptools

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="rvfln",
    version="0.0.5",
    author="Márton Kardos",
    description="Package for an implementation of Random Vector Functional Link Network. For the user's convenience the package uses Sklearn's API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["rvfln"],
)
