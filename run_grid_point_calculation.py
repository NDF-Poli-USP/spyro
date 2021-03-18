import sys
sys.path.append('/home/alexandre/Development/Spyro-main/spyro')
import spyro

print("===================================================", flush = True)
frequency = 5.0
method = 'KMV'
degree = 3
print("Running with "+method+ " and p =" + str(degree), flush = True)

G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.2, G_init= 12)

print("===================================================", flush = True)

print('final G', flush = True)
print(G)
# print("===================================================", flush = True)
# print("===================================================", flush = True)

# frequency = 5.0
# method = 'KMV'
# degree = 3
# print("Running with "+method+ " and p =" + str(degree), flush = True)

# G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.2, G_init= 12)

# print("===================================================", flush = True)
# print('final G', flush = True)
# print(G, flush = True)
# print("===================================================", flush = True)
# print("===================================================", flush = True)


# frequency = 5.0
# method = 'KMV'
# degree = 4
# print("Running with "+method+ " and p =" + str(degree), flush = True)

# G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.2, G_init= 12)

# print("===================================================", flush = True)
# print('final G', flush = True)
# print(G, flush = True)
# print("===================================================", flush = True)