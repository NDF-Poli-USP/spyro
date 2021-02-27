from setuptools import setup, find_packages

setup(
    name="spyro",
    version="0.0.16",
    license="GPL v3",
    description="acoustic wave modeling with the finite element method",
    author="João A. Isler,  Keith J. Roberts, Alexandre F. G. Olender, Lucas Franceschini",
    url="https://github.com/krober10nd/spyro",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "exdown","segyio"],
)
