"""
Setup configuration for rlcard-gin-rummy-bot.
"""

from setuptools import setup, find_packages

setup(
    name="rlcard-gin-rummy-bot",
    version="0.1.0",
    description="Autonomous High-Performance RLCard Gin Rummy Bot with Vectorized PPO, Multi-Agent Self-Play, and PyTorch CUDA Acceleration",
    long_description=open("README.md", "r", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Reinforcement Learning Team",
    author_email="team@antigravity.ai",
    packages=find_packages(include=["gin_rummy", "gin_rummy.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "rlcard>=1.2.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.0.0",
            "black>=23.7.0",
            "isort>=5.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gin-train=scripts.train:main",
            "gin-benchmark=scripts.benchmark:main",
            "gin-mcp=scripts.run_mcp_workflow:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
)
