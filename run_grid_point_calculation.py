import sys
sys.path.append('/home/alexandre/Development/Spyro-main/spyro')
import spyro

print("===================================================")
frequency = 5.0
method = 'KMV'
degree = 4
print("Running with "+method+ " and p =" + str(degree))

G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.5, G_init= 12)

print("===================================================")

print('final G')
print(G)
# print("===================================================")
# print("===================================================")
# print("Running with KMV and p =3")

# frequency = 5.0
# method = 'KMV'
# degree = 3

# G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.1)

# print("===================================================")
# print('final G')
# print(G)
# print("===================================================")
# # print("===================================================")
# # print("Running with KMV and p =4")

# # frequency = 5.0
# # method = 'KMV'
# # degree = 4

# G = spyro.tools.minimum_grid_point_calculator(frequency, method, degree, experient_type = 'homogeneous', TOL = 0.1)

# print("===================================================")
# print('final G')
# print(G)
# print("===================================================")