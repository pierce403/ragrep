"""Setup script for RAGrep."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

extras = {
    "retrieval": [
        "chromadb>=0.4.15",
    ],
    "generation": [
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "torch>=2.0.0; python_version<'3.12'",
    ],
}
extras["full"] = extras["retrieval"] + extras["generation"]

setup(
    name="ragrep",
    version="0.1.0",
    author="RAGrep Team",
    description="AI Agent File Navigator - Semantic search tool similar to grep for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "ragrep=ragrep.cli:main",
        ],
    },
)
