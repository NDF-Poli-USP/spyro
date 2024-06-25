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
python3 -m pytest --maxfail=1 test/
````
