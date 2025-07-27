"""
NIDS Autoencoder Project Setup Configuration
============================================

A production-ready Network Intrusion Detection System using autoencoders
for anomaly detection in network traffic.

Author: Your Name
Email: your.email@example.com
License: MIT
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

from setuptools import find_packages, setup

# ============ Package Information ============
PACKAGE_NAME = "nids_autoencoder"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"
DESCRIPTION = "Production-ready Network Intrusion Detection System using Autoencoders"
LICENSE = "MIT"
URL = "https://github.com/yourusername/nids-autoencoder"
DOWNLOAD_URL = "https://github.com/yourusername/nids-autoencoder/archive/v1.0.0.tar.gz"

# ============ Project Paths ============
HERE = Path(__file__).parent.absolute()
README_PATH = HERE / "README.md"
REQUIREMENTS_PATH = HERE / "requirements.txt"
REQUIREMENTS_DEV_PATH = HERE / "requirements-dev.txt"

# ============ Read Files ============
def read_file(file_path: Path) -> str:
    """Read file content safely."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def read_requirements(file_path: Path) -> List[str]:
    """Read requirements from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        requirements = []
        for line in lines:
            line = line.strip()
            # Skip comments, empty lines, and -r references
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
        return requirements
    except FileNotFoundError:
        return []

# ============ Long Description ============
long_description = read_file(README_PATH)
if not long_description:
    long_description = DESCRIPTION

# ============ Requirements ============
install_requires = read_requirements(REQUIREMENTS_PATH)
dev_requires = read_requirements(REQUIREMENTS_DEV_PATH)

# ============ Classifiers ============
classifiers = [
    # Development Status
    "Development Status :: 5 - Production/Stable",
    
    # Intended Audience
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: System Administrators",
    "Intended Audience :: Information Technology",
    
    # Topic
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Security",
    "Topic :: System :: Networking :: Monitoring",
    "Topic :: Software Development :: Libraries :: Python Modules",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Operating System
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    
    # Programming Language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    
    # Framework
    "Framework :: FastAPI",
    "Framework :: Jupyter",
    
    # Environment
    "Environment :: Console",
    "Environment :: Web Environment",
    
    # Natural Language
    "Natural Language :: English",
]

# ============ Keywords ============
keywords = [
    "network-security",
    "intrusion-detection",
    "anomaly-detection",
    "autoencoder",
    "machine-learning",
    "deep-learning",
    "cybersecurity",
    "network-monitoring",
    "production-ready",
    "mlops",
]

# ============ Project URLs ============
project_urls = {
    "Homepage": URL,
    "Documentation": f"{URL}/docs",
    "Repository": URL,
    "Bug Reports": f"{URL}/issues",
    "Source": URL,
    "Download": DOWNLOAD_URL,
    "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    "Contributing": f"{URL}/blob/main/CONTRIBUTING.md",
    "Security": f"{URL}/blob/main/SECURITY.md",
}

# ============ Entry Points ============
entry_points = {
    "console_scripts": [
        "nids-train=src.cli.train:main",
        "nids-predict=src.cli.predict:main",
        "nids-serve=src.cli.serve:main",
        "nids-evaluate=src.cli.evaluate:main",
        "nids-monitor=src.cli.monitor:main",
    ],
}

# ============ Package Data ============
package_data = {
    "nids_autoencoder": [
        "config/*.yaml",
        "config/*.yml",
        "config/*.json",
        "templates/*.html",
        "static/*",
        "data/sample_data.csv",
    ],
}

# ============ Data Files ============
data_files = [
    ("config", ["config/default.yaml", "config/production.yaml"]),
    ("scripts", ["scripts/setup.sh", "scripts/deploy.sh"]),
    ("docs", ["README.md", "CHANGELOG.md", "LICENSE"]),
]

# ============ Extras Require ============
extras_require = {
    "dev": dev_requires,
    "test": [
        "pytest>=7.2.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-asyncio>=0.21.0",
    ],
    "docs": [
        "sphinx>=6.1.0",
        "sphinx-rtd-theme>=1.2.0",
        "mkdocs>=1.4.0",
        "mkdocs-material>=9.0.0",
    ],
    "monitoring": [
        "prometheus-client>=0.16.0",
        "grafana-api>=1.0.3",
        "structlog>=22.3.0",
    ],
    "cloud": [
        "boto3>=1.26.0",
        "google-cloud-storage>=2.7.0",
        "azure-storage-blob>=12.14.0",
    ],
    "serving": [
        "fastapi>=0.95.0",
        "uvicorn[standard]>=0.20.0",
        "gunicorn>=20.1.0",
    ],
    "all": dev_requires,
}

# ============ Python Requires ============
python_requires = ">=3.8,<3.12"

# ============ Setup Configuration ============
setup_kwargs = {
    "name": PACKAGE_NAME,
    "version": VERSION,
    "author": AUTHOR,
    "author_email": AUTHOR_EMAIL,
    "description": DESCRIPTION,
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "url": URL,
    "download_url": DOWNLOAD_URL,
    "project_urls": project_urls,
    "packages": find_packages(where=".", exclude=["tests*", "docs*", "scripts*"]),
    "package_dir": {"": "."},
    "package_data": package_data,
    "data_files": data_files,
    "include_package_data": True,
    "install_requires": install_requires,
    "extras_require": extras_require,
    "python_requires": python_requires,
    "entry_points": entry_points,
    "classifiers": classifiers,
    "keywords": ", ".join(keywords),
    "license": LICENSE,
    "platforms": ["any"],
    "zip_safe": False,
}

# ============ Setup Execution ============
if __name__ == "__main__":
    # Validate Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required.")
        sys.exit(1)
    
    # Validate required files
    required_files = [REQUIREMENTS_PATH]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print(f"ERROR: Required files missing: {missing_files}")
        sys.exit(1)
    
    # Run setup
    setup(**setup_kwargs)

# ============ Development Commands ============
"""
Common development commands:

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with all extras
pip install -e .[all]

# Install for testing
pip install -e .[test]

# Install for documentation
pip install -e .[docs]

# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*

# Clean build artifacts
python setup.py clean --all
rm -rf build/ dist/ *.egg-info/
"""
