from setuptools import setup, find_packages

setup(
    name="torchtmpl",
    version="0.0.1",
    packages=find_packages(where="."),  # Automatically find Python packages
    install_requires=[
        "PyYAML>=6.0",
        "torch>=1.13.1",
        "torchvision>=0.14.1",
        "torchinfo>=1.7.2",
        "wandb>=0.13.11",
        "scipy>=1.10.1",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.1",
    ],
    python_requires=">=3.7",  # Specify minimum Python version
    description="A PyTorch template project",
    author="Your Name",
    author_email="your.email@example.com",
)
