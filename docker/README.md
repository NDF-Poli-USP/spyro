## Apptainer images

Apptainer is a container platform designed to run complex applications on HPC clusters. The steps below were tested in Ubuntu 24.04.1 LTS.

### Installing Apptainer

Apptainer may be installed with the following instructions (there are issues with previous versions in Ubuntu 24):
````
sudo apt update
sudo apt install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.3.6/apptainer_1.3.6_amd64.deb
sudo apt install -y ./apptainer_1.3.6_amd64.deb
````

### Building Apptainer images

In spyro, the Apptainer image is based on the Docker image built below. Converting from Docker to Apptainer is more productive because Apptainer does not support image layers by design. Then, it is necessary to rerun all the instructions of the definition file when creating Apptainer images even when just one instruction is added for testing purposes.
````
export IMAGE_ID=$(sudo docker images -q devtag:1.0)
sudo docker save $IMAGE_ID -o devimg.tar
sudo chown $USER:$USER devimg.tar
apptainer build devimg.sif docker-archive://devimg.tar
````

### Running Apptainer images

To execute the image in interactive mode:
````
apptainer shell --bind /tmp/cache-${USER}:/home/firedrake/firedrake/.cache -e devimg.sif
````

To execute the image in batch mode:
````
mkdir /tmp/cache-${USER}
apptainer run --bind /tmp/cache-${USER}:/home/firedrake/firedrake/.cache -e devimg.sif ./workshop_script.sh
````

## Docker images

This directory presents a Dockerfile to provide images for release (user installation), and development/testing. The release image comes with the corresponding spyro version installed, while the development/testing image use the current spyro checkout. The latest spyro version is not installed in the development/testing image to avoid ambiguities during use.

The build commands below assume that the working directory is the root of spyro. If your current working directory is different, you have to adjust the Dockerfile path.

### Installing

The Docker image may be used for installing spyro. The following command builds the release image:
````
docker build -t runtag:1.0 --target spyro_release docker
````

Then, the following commands gives access to a virtual environment with spyro:
````
docker run -it runtag:1.0
. firedrake/bin/activate
````

### Development/Testing

The Dockerfile may also be used to create a development environment. First, clone the git repository and then build the development image:
````
git clone https://github.com/Olender/spyro-1.git
cd spyro-1
git checkout <your_branch>
docker build -t devtag:1.0 -f docker/Dockerfile --target spyro_development docker
````

Then, start a container and share your local repository:
````
docker run -v $PWD:/home/firedrake/shared/spyro -it devtag:1.0
. firedrake/bin/activate
````

For running the automated tests:
````
cd shared/spyro/
python3 -m pytest test/
python3 -m pytest test_3d/
mpiexec -n 6 python3 -m pytest test_parallel/
````
