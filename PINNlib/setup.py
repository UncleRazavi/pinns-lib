from setuptools import setup, find_packages

setup(
    name="pinns-lib",
    version="0.1.0",
    description="Physics-Informed Neural Networks library",
    author="H.R.",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib"
    ],
)
