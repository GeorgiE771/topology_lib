from setuptools import setup, find_packages

setup(
    name="topology_lib",
    version="0.1.0",
    author="Evgeniy",
    author_email="uimsasd@gmail.com",
    description="A library for topological preservation in autoencoders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GeorgiE771/topology_lib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.11',
    install_requires=[
        "torch",
        "ripser",
        "numpy"
    ],
)