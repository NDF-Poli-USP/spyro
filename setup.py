from setuptools import setup, find_packages

setup(
    name="spyro",
    version="0.0.20",
    license="GPL v3",
    description="acoustic wave modeling with the finite element method",
    author="Keith J. Roberts, Alexandre F. G. Olender, Lucas Franceschini",
    url="https://github.com/krober10nd/spyro",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "exdown==0.7.0", "segyio", "SeismicMesh"],
)
