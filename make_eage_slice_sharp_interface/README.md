You have to download the EAGE velocity model to run these scripts. 

wget https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/Salt_Model_3D.tar.gz

This model this is sliced in these scripts to produce the true velocity model.

1. Build the velocity models by running make_eage_slice_velocity_models.py

2. Run the mesh generation and input file creation scripts by running make_eage_slice_meshes.py

3. Put the meshes (*.msh) in the meshes folder `cp *.msh ../meshes` and the velocity models `cp *.hdf5 ../velocity_models` in the velocity model folder. 

You'll need `SeismicMesh >= 3.5.0` with `segyio` 
