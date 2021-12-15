import spyro
from spyro.utils.synthetic import smooth_field
from SeismicMesh import write_velocity_model
import os

# try 100:
input = "velocity_models/vp_marmousi-ii.segy"
output = "velocity_models/vp_marmousi-ii_smoother_guess.segy"

smooth_field(input, output, show = True, sigma =400)

vp_filename, vp_filetype = os.path.splitext(output)

write_velocity_model(output, ofname = vp_filename)

