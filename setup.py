from setuptools import setup, find_packages

setup(
    name="demand-forecasting-tft",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytorch-forecasting>=1.0.0",
        "pytorch-lightning>=2.0.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    author="Malhar Shinde",
    description="Interpretable demand forecasting using Temporal Fusion Transformers",
    python_requires=">=3.8",
)
