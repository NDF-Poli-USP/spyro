FROM firedrakeproject/firedrake:2024-05 AS spyro_base

USER root
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && apt-get -y install gmsh patchelf \
    && rm -rf /var/lib/apt/lists/*

USER firedrake
RUN . /home/firedrake/firedrake/bin/activate; pip3 install wheel --upgrade
RUN . /home/firedrake/firedrake/bin/activate; pip3 install scipy
RUN . /home/firedrake/firedrake/bin/activate; pip3 install roltrilinos
RUN . /home/firedrake/firedrake/bin/activate; pip3 install ROL

# spyro dependencies
USER root
# Is there a reason for using development versions?
RUN apt-get update && apt-get install -y libgmp3-dev libmpfr-dev libcgal-dev
USER firedrake
RUN . /home/firedrake/firedrake/bin/activate; pip3 install matplotlib
RUN . /home/firedrake/firedrake/bin/activate; pip3 install segyio
RUN . /home/firedrake/firedrake/bin/activate; pip3 install --no-dependencies SeismicMesh[io]
RUN . /home/firedrake/firedrake/bin/activate; pip3 install pyamg
RUN . /home/firedrake/firedrake/bin/activate; pip3 install meshio

FROM spyro_base AS spyro_release

RUN . /home/firedrake/firedrake/bin/activate; pip install git+https://github.com/Olender/spyro-1.git
