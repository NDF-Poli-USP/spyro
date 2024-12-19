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

In spyro, the Apptainer image is based on the [Docker image](../docker/). Converting from Docker to Apptainer is more productive because Apptainer does not support image layers by design. Then, it is necessary to rerun all the instructions of the definition file when creating Apptainer images even when just one instruction is added for testing purposes.
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