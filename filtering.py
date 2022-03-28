import spyro
from spyro.utils.synthetic import smooth_field
from SeismicMesh import write_velocity_model
import os

# try 100:
sigma = 400
input = "velocity_models/cut_marmousi.segy"
output = "velocity_models/cut_marmousi_"+str(sigma)+".segy"

smooth_field(input, output, show = True, sigma =sigma)

vp_filename, vp_filetype = os.path.splitext(output)

write_velocity_model(output, ofname = vp_filename)

