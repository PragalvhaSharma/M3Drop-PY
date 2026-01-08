import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="M3Drop",
    version="0.4.1",  # <--- BUMP VERSION to fix the crash
    author="Tallulah Andrews, Anthony Son, Pragalvha Sharma",
    author_email="tandrew6@uwo.ca, json59@uwo.ca, pragalvhasharma@gmail.com",
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
        "anndata>=0.8.0",          
        "h5py>=3.8.0",             
        "matplotlib>=3.5.0",       # Relaxed: Fixes the Supercomputer crash
        "matplotlib-venn>=0.11",
        "memory_profiler>=0.60.0",
        "numpy>=1.21.0",           
        "pandas>=1.5.0",           # Relaxed: Fixes the dependency conflict
        "scanpy>=1.9.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.8.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.0",
    ],
    # --- ADD THIS BLOCK FOR CUPY ---
    extras_require={
        "gpu": ["cupy-cuda12x"],  # Users can install this optionally
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
