ARG FIREDRAKE_VERSION=2025.4.1
FROM firedrakeproject/firedrake:${FIREDRAKE_VERSION} AS spyro_base

USER root
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install segyio

# spyro dependencies
USER root
RUN apt-get update && apt-get install -y libgmp3-dev libmpfr-dev libcgal-dev python3-tk

FROM spyro_base AS spyro_release

RUN pip install --no-cache-dir git+https://github.com/NDF-Poli-USP/spyro.git

FROM spyro_base AS spyro_development

WORKDIR /workspace
