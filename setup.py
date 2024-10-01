import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simpl",  
    version="0.0.0", 
    author="",
    description="simpl: a python package for optimising neural representations from behavioural initialisation using an EM-style algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="", 
    packages=setuptools.find_packages(), # finds all packages inside the parent directory containing a __init__.py file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'simpl': ['data/*'],
    },
    install_requires=[
        'jax',
        'xarray',
        'kalmax',
        'scikit-learn',
        'scikit-image',
        'matplotlib',
        'tqdm',
        'netcdf4',
        'h5netcdf',
    ],
    extras_require={
        'demo': ['jupyter'],
    }
)


