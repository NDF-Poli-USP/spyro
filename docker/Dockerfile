ARG FIREDRAKE_VERSION=2025.4.1
FROM firedrakeproject/firedrake:${FIREDRAKE_VERSION} AS spyro_base

USER root
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install segyio

USER root
RUN apt-get update && apt-get install -y libgmp3-dev libmpfr-dev libcgal-dev python3-tk
RUN pip3 install --no-cache-dir  meshio
RUN pip3 install --no-cache-dir  pyamg

FROM spyro_base AS spyro_release

RUN pip install --no-cache-dir git+https://github.com/NDF-Poli-USP/spyro.git

FROM spyro_base AS spyro_development
# spyro dependencies
# I install some here instead of the devcontainers.json 
# to reduce time rebuilding containers in vs code

RUN pip3 install --no-dependencies --no-cache-dir git+https://github.com/NDF-Poli-USP/SeismicMesh.git
RUN pip3 install --no-cache-dir "pakmsh[examples] @ git+https://github.com/NDF-Poli-USP/PakMsh.git"
RUN pip3 install -U memory_profiler

WORKDIR /workspace
