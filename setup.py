from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="xtars",
    version="0.1",
    description="Zero/few-shot text classification for large label sets",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Angelo Ziletti",
    author_email="angelo.ziletti@gmail.com",
    url="https://github.com/Bayer-Group/xtars",
    packages=find_packages(exclude="tests"),  # same as name
    license="",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.6",
)
