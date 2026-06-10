from setuptools import setup, find_packages

setup(
    name="fpga-event-drone",
    version="1.0.0",
    author="Enotrium",
    author_email="dev@enotrium.com",
    description="Real-time obstacle detection and evasion for drones using event cameras, normal flow estimation, and FPGA-accelerated inference.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Enotrium/FPGA-Event-Based-encode",
    packages=find_packages(),
    include_package_data=True,
    package_data={"models": ["models/*.pth"],
                  "models.models": ["*.pth"]},
    install_requires=[
        "torch>=1.13.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "tqdm>=4.64",
        "matplotlib>=3.7",
        "matplotlib-inline>=0.1",
        "opencv-python>=4.8",
        "pyyaml>=6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)