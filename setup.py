import os

from setuptools import find_packages, setup

# 获取当前目录的绝对路径
current_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name="thesis",
    packages=find_packages(),
    install_requires=[f"BEAR @ file://{current_dir}/BEAR"],
)
