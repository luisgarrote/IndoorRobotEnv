from setuptools import setup, find_packages

setup(
    name="indoor_robot_2025",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
    ],
    description="Indoor robot navigation environment for Deep RL (2025 edition).",
    author="YOUR NAME",
)
