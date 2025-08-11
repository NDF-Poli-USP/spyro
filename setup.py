from setuptools import setup, find_packages

setup(
    name="spyro",
    version="0.9.1",
    license="LGPL v3",
    description="Wave modeling with the finite element method",
    author="Keith J. Roberts, Alexandre F. G. Olender, Ruben Andres Salas, Eduardo Moscatelli de Souza, Daiane I. Dolci, Thiago Dias dos Santos, Lucas Franceschini",
    url="https://github.com/NDF-Poli-USP/spyro",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "segyio",
        "meshio",
        "mpi-pytest"],
)
