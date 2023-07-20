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

#  x(m), y(m), c(km/s) ,t(ms)
# 9.200000000000000000e+03	1.687500000000000000e+03	3.500000000000000444e+00	1.122692634216587066e+03
# 5.662500000000000000e+03	0.000000000000000000e+00	3.859560000000000102e+00	1.039170607108715558e+03

Wave.forward_solve()

print("END")