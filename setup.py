from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="franka_gym",
    version="0.1.0",
    description="Gymnasium environment for Franka robot control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Baris Yazici",
    author_email="barisyazici@sabanciuniv.edu",
    url="https://github.com/barisyazici/franka_gym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "stable-baselines3>=2.0.0",
        # "franka_bindings",  # Your local franka bindings package
    ],
    python_requires=">=3.8",
    include_package_data=True,
) 