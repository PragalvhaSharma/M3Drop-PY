import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="M3Drop",
    version="0.1.2",
    author="Pragalvha Sharma, Tallulah Andrews",
    author_email="pragalvhasharma@gmail.com, tandrew6@uwo.ca",
    description="A Python implementation of the M3Drop single-cell RNA-seq analysis tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PragalvhaSharma/m3DropNew",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "anndata==0.10.9",
        "matplotlib==3.9.4",
        "matplotlib-venn==1.1.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scanpy==1.10.3",
        "scikit-learn==1.4.2",
        "scipy==1.13.0",
        "seaborn==0.13.2",
        "statsmodels==0.14.4",
    ],
)
