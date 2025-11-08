"""
Como usar para escolher o número de arquivos para abrir por vez

 PV_START=0 PV_END=10 paraview --state=paraview_open_transform.py
"""


# state file generated using paraview version 5.11.0-RC1
from paraview.simple import *
import paraview
import os
import glob
from math import ceil

paraview.compatibility.major = 5
paraview.compatibility.minor = 11

# Definir o diretório de trabalho como o diretório onde o script está localizado
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Domain dimensions
h = 120
l = 316

# import the simple module from the paraview
# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.InteractionMode = '2D'
renderView1.StereoType = 'Crystal Eyes'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1510, 816)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# Informações que se repetem
point_arrays = ['beta_f', 'G3', 'G4' 'DLa_p', 'DLa_rho', 'DLa_beta',
                'p_f', 'rho_f', 'rho_phys', 'T', 'u', 'von Mises']

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# Procurando pelos arquivos *.xdmf
xdmf_files = glob.glob(
    '{}/**/*fields.xdmf'.format(script_dir), recursive=True)
xdmf_files.sort()

# Usar os argumentos de linha de comando
start = int(os.getenv('PV_START', 0))
end = int(os.getenv('PV_END', 40))

# Calculate the number of results in columns and rows
n = end - start
ny = ceil((3 / 4 * n * l / h)**0.5)
nx = ceil(n / ny)

print('n:', n, 'nx: ', nx, 'ny: ', ny)
print('Range:', start, 'to', end)

# Initializate the positions
x = 0
y = 0

# Calculate the space
d = min(h, l) * 5. / 100

# Iniciar contadores
ix = 0
iy = 0

print('Creating ...')
for xdmf_file in xdmf_files[start:end]:
    print(xdmf_file)

    # create a new 'Xdmf3ReaderS'
    fields_xdmf = Xdmf3ReaderS(
        registrationName='fields_.xdmf', FileName=[xdmf_file])
    fields_xdmf.PointArrays = point_arrays
    # create a new 'Transform'
    transform = Transform(registrationName='Transform', Input=fields_xdmf)
    transform.Transform = 'Transform'

    # init the 'Transform' selected for 'Transform'
    transform.Transform.Translate = [x + d * ix, y + d * iy, 0.0]

    # create a new 'Reflect' filter for the mirrored version
    reflect = Reflect(registrationName='Reflect', Input=transform)
    reflect.Plane = 'X Max'

    # Mostrar o campo "Physical Pseudo-densities"
    display = Show(reflect, renderView1)
    display.SetRepresentationType('Surface')
    display.ColorArrayName = ['CELLS', 'rho_phys']
    ColorBy(display, ('CELLS', 'rho_phys'))
    display.RescaleTransferFunctionToDataRange(True, False)
    display.SetScalarBarVisibility(renderView1, True)

    # Update x
    x += l

    # Update x counter
    ix += 1

    if ix == nx:
        x = 0
        ix = 0
        y += h
        iy += 1

if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
