from setuptools import setup, find_packages

setup(
    name="sae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "scikit-learn",
        "pandas",
    ],
    python_requires=">=3.8",
    description="Sparse Autoencoder and Probing Tools",
    author="Barak",
)
