from setuptools import setup, find_packages

setup(
    name="unified-devops-nexus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "azure-identity>=1.12.0",
        "azure-mgmt-resource>=23.0.1",
        "azure-mgmt-monitor>=5.0.0",
        "pytest>=7.4.0",
        "python-dotenv>=1.0.0",
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