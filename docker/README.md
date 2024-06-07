## Docker images

### Installing

The Docker image may be used for installing spyro. The following command builds the release image:
````
docker build -t runtag:1.0 --target spyro_release .
````

Then, the following commands gives access to a virtual environment with spyro:
````
docker run -it runtag:1.0
. firedrake/bin/activate
````

### Testing

The Docker image for running the tests is built with the following command (considering that your working directory is the same where the Dockerfile is located):
````
docker build -t testtag:1.0 --target spyro_test .
````

Then, the following command may be called for running the tests:
````
docker run -it testtag:1.0 /bin/bash -c "source /home/firedrake/firedrake/bin/activate; python3 -m pytest --maxfail=1 ."
````

### Development

The Dockerfile may also be used to create a development environment. First, clone the git repository and then build the development image:
````
git clone https://github.com/Olender/spyro-1.git
cd spyro-1
git checkout <your_branch>
docker build -t devtag:1.0 -f docker/Dockerfile --target spyro_development .
````

Then, start a container and share your local repository:
````
docker run -v $PWD/spyro:/home/firedrake/shared/spyro -it devtag:1.0
. firedrake/bin/activate
````
