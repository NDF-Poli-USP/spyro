## Apptainer images

Apptainer is a container platform designed to run complex applications on HPC clusters. The steps below were tested in Ubuntu 24.04.1 LTS.

### Installing Apptainer

Apptainer may be installed with the following instructions (there are issues with previous versions of Apptainer in Ubuntu 24):
````
sudo apt update
sudo apt install -y wget
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.3.6/apptainer_1.3.6_amd64.deb
sudo apt install -y ./apptainer_1.3.6_amd64.deb
````

### Building Apptainer images

In spyro, the Apptainer image is based on the [Docker image](../docker/). Converting from Docker to Apptainer is more productive because Apptainer does not support image layers by design. Then, it is necessary to rerun all the instructions of the definition file when creating Apptainer images even when just one instruction is added (for example, for testing purposes).
````
export IMAGE_ID=$(sudo docker images -q devtag:1.0)
sudo docker save $IMAGE_ID -o devimg.tar
sudo chown $USER:$USER devimg.tar
sudo apptainer build devimg.sif docker-archive://devimg.tar
````

### Running Apptainer images

Before running the Apptainer image, it is necessary to create an [overlay filesystem](https://docs.sylabs.io/guides/3.6/user-guide/persistent_overlays.html) because the SIF container is read only and spyro requires writing to the filesystem.
````
apptainer overlay create --size 1024 /tmp/ext3_overlay.img
````

To execute the image in interactive mode:
````
apptainer shell --overlay /tmp/ext3_overlay.img -e devimg.sif
````

To execute the image in batch mode:
````
apptainer run  --overlay /tmp/ext3_overlay.img -e devimg.sif <script>
````