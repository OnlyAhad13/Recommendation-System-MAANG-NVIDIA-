"""
Setup script for the recommendation system package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="recommendation-system",
    version="1.0.0",
    author="Syed Abdul Ahad",
    author_email="syedahad171@gmail.com",
    description="Enterprise-scale recommendation system with two-tower retrieval and DCN ranking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OnlyAhad13/recommendation-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Recommender Systems :: Recommendation System",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.13.0",
        "tensorflow-recommenders>=0.7.0",
        "numpy>=1.23.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "psutil>=5.9.0",
        "faiss-cpu>=1.7.4",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "gpu": ["faiss-gpu>=1.7.4"],
    },
    entry_points={
        "console_scripts": [
            "recsys-train=scripts.train:main",
            "recsys-preprocess=scripts.preprocess:main",
        ],
    },
)

