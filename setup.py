import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="harmonic_inference",
    version="0.0.1",
    author="Andrew McLeod",
    author_email="andrew.mcleod@epfl.ch",
    description="A package to perform harmonic inference on various data structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apmcleod/harmonic_inference",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
