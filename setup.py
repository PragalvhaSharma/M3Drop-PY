import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="M3Drop",
    version="0.2.2",
    author="Tallulah Andrews, Pragalvha Sharma, Anthony Son",
    author_email="tandrew6@uwo.ca, pragalvhasharma@gmail.com, json59@uwo.ca",
    description="A Python implementation of the M3Drop single-cell RNA-seq analysis tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PragalvhaSharma/m3DropNew",
    license="MIT",
    packages=setuptools.find_packages(include=["m3Drop", "m3Drop.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.8',
    install_requires=[
        "anndata==0.10.9",
        "matplotlib==3.9.4",
        "matplotlib-venn==1.1.2",
        "memory_profiler==0.61.0",
        "numpy>=2.0,<3",          # was 1.26.4 (no py3.13 wheels)
        "pandas>=2.2.3,<2.3",     # was 2.2.2
        "scanpy==1.10.3",
        "scikit-learn==1.7.1",
        "scipy>=1.14.1,<1.15",    # was 1.13.0
        "seaborn==0.13.2",
        "statsmodels==0.14.4",
    ],
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
