import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rnem",  
    version="0.0.0", 
    author="Tom George",
    description="rNEM: a python package for optimising neural representations using relaxed neural EM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomGeorge1234/rNEM", 
    packages=setuptools.find_packages(), # finds all packages inside the parent directory containing a __init__.py file
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'rnem': ['data/*'],
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
        'demo': [],
    }
)


