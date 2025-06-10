from setuptools import find_packages, setup

setup(
    name="BEAR",
    packages=find_packages(),
    install_requires=[
        "sb3_contrib>=2.0.0a1",
        "stable_baselines3>=2.0.0a1",
        "pvlib",
        "scikit-learn",
        "cvxpy",
        "tqdm",
        "openai",
    ],
)
