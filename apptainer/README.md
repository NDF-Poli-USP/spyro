## Apptainer images

Apptainer is a container platform designed to run complex applications on HPC clusters. The steps below were tested in Ubuntu 24.04.1 LTS and in USP's Mintrop HPC cluster.

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

In spyro, the Apptainer image is based on the [Docker image](../docker/). Converting from Docker to Apptainer is more productive because Apptainer does not support image layers by design. Then, it is necessary to rerun all the instructions of the definition file when creating Apptainer images even when just one instruction is added (for example, for testing purposes). Because of the sudo command you'll need to do this locally in your PC even if your end goal is running code in Mintrop.
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

### Running on Mintrop
In order to run code with Apptainer on you first need to send your container image to Mintrop, we recommend using rsync with a command similar to the one below (but with your desired file path) in your local PC to send the sif image:
````
rsync -avP --info=progress2 $PATH_LOCAL$/*.sif $USER$@200.144.186.101:$PATH_MINTROP$
````
If you are using the same Firedrake version as spyro's main branch you can also copy this image from Mintrop's public/apptainer/ folder.

In order to run code with Apptainer on Mintrop, you may add the following lines to your Slurm script:
````
module load apptainer
apptainer overlay create --size 1024 /tmp/ext3_overlay.img
apptainer run  --overlay /tmp/ext3_overlay.img -e devimg.sif ./example.sh
````

where the script [example.sh](./example.sh) is provided in this directory to show how spyro code could be called.
