import spyro

dictionary = {}
dictionary["acquisition"] = {
    "source_locations": [(-0.1, 0.3),(-0.1, 0.6)],
}

Wave = spyro.examples.Rectangle(model_dictionary=dictionary)

Wave.forward_solve()

print("END")


