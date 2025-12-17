from setuptools import setup

setup(
    name="pinns-lib",
    version="0.1.1",
    description="Physics-Informed Neural Networks library",
    author="H.R.",
    py_modules=["PDEproblem", "PINNModel", "trainer", "utils", "visualizer"],
    install_requires=["torch", "numpy", "matplotlib"],
)
