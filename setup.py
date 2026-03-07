from setuptools import setup, find_packages

setup(
    name="mevlstp",
    version="1.0.0",
    description="MEVL-STP: Multi-Encoder and Vision Language Model for Arbitrarily Shaped Scene Text Spotting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.45.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
        "shapely>=2.0.0",
    ],
)
