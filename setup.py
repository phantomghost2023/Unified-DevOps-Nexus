from setuptools import setup, find_packages

setup(
    name="unified-devops-nexus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "pytest>=8.0.0",
        "pytest-cov>=6.0.0",
        "pytest-asyncio>=0.23.0",
        "pytest-mock>=3.14.0",
        "typing-extensions>=4.0.0"
    ],
    python_requires=">=3.8",
    description="Unified DevOps Infrastructure as Code Framework",
    author="DevOps Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.8",
    ]
)