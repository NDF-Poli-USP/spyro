from setuptools import setup, find_packages

setup(
    name="spyro",
    version="0.9.0",
    license="LGPL v3",
    description="Wave modeling with the finite element method",
    author="Keith J. Roberts, Alexandre F. G. Olender, Lucas Franceschini, Eduardo Moscatelli de Souza, Daiane I. Dolci",
    url="https://github.com/NDF-Poli-USP/spyro",
    packages=find_packages(),
    install_requires=[
        "firedrake",
        "numpy",
        "scipy",
        "matplotlib",
        "exdown==0.7.0",
        "segyio",
        "meshio"],
)
