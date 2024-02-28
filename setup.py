import json
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Get the long description from the README file
ROOT_DIR = Path(__file__).parent.resolve()
long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
VERSION_FILE = ROOT_DIR / "backoffice" / "VERSION"
VERSION = json.loads(VERSION_FILE.read_text(encoding="utf-8"))["version"]

if sys.version_info < (3, 9):
    sys.exit("backoffice requires Python >= 3.9")

setup(
    name="backoffice",
    version=VERSION,
    description="backoffice to control bioimage.io collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bioimage-io/collection",
    author="bioimage.io Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "bioimageio.core @ git+https://github.com/bioimage-io/core-bioimage-io-python@40aae1005e6caea4a3ad6104f54deba8582d0f87",  # TODO: change to released version
        "bioimageio.spec @ git+https://github.com/bioimage-io/spec-bioimage-io@b9714ba8d8214c67e9377a8cd24f6ba65ed1b981",  # TODO: change to released version
        "fire",
        "loguru",
        "minio==7.2.3",
        "ruyaml",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "black",
            "pdoc",
            "pre-commit",
            "pyright",
            "pytest",
            "torch",
        ]
    },
    entry_points={"console_scripts": ["backoffice = backoffice.__main__:main"]},
    project_urls={
        "Bug Reports": "https://github.com/bioimage-io/collection/issues",
        "Source": "https://github.com/bioimage-io/collection",
    },
)
