import spyro
dictionary = {}
dictionary["visualization"] = {
    "forward_output" : True,
    "output_filename": "results/forward_output.pvd",
    "fwi_velocity_model_output": False,
    "velocity_model_filename": None,
    "gradient_output": False,
    "gradient_filename": None,
}

Wave = spyro.examples.Cut_marmousi_acoustic(model_dictionary=dictionary)

Wave.forward_solve()

print("END")