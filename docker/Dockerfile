FROM firedrakeproject/firedrake AS spyro_base

USER root
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install segyio

# spyro dependencies
USER root
RUN apt-get update && apt-get install -y libgmp3-dev libmpfr-dev libcgal-dev python3-tk
RUN pip3 install --no-dependencies git+https://github.com/NDF-Poli-USP/SeismicMesh.git
RUN pip3 install pyamg
RUN pip3 install meshio

FROM spyro_base AS spyro_release

RUN pip install git+https://github.com/NDF-Poli-USP/spyro.git

FROM spyro_base AS spyro_development

# Profiling tools
RUN pip3 install -U memory_profiler

RUN pip3 install pytest pytest-cov
WORKDIR /home/
RUN echo "/home/shared/spyro" >> /usr/local/lib/python3.12/site-packages/shared.pth
