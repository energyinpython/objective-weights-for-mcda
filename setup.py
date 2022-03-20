import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="objective-weights-mcda",
    version="0.0.5",
    author="Aleksandra Ba",
    author_email="aleksandra.baczkiewicz@phd.usz.edu.pl",
    description="Package for Multi-Criteria Decision Analysis with Objective Criteria Weighting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/energyinpython/objective-weights-for-mcda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)