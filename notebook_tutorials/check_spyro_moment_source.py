"""Verifica a fonte de momento nativa do Spyro (amplitude = matriz identidade).

Ideia: uma fonte explosiva correta aplicada a um campo de deslocamento u
devolve div(u)(x_s). Para campos LINEARES o gradiente é constante, então o
resultado não depende da célula escolhida (mesmo com a fonte num vértice):

    u = (z, 0)  ->  div u = 1
    u = (0, x)  ->  div u = 1

Fonte correta  =>  os dois valores saem IGUAIS (a mesma escala).
Sinais opostos ou magnitudes diferentes  =>  a tabulação de ordem 1 do
Spyro não está mapeando as derivadas para o elemento físico.

Uso (no script de forward, modo "displacement", ANTES de criar o Wave_obj):
    dictionary["acquisition"]["amplitude"] = np.eye(2)
e, depois de criar o Wave_obj:
    from check_spyro_moment_source import check_spyro_moment_source
    check_spyro_moment_source(Wave_obj)
"""
import numpy as np
import firedrake as fire


def check_spyro_moment_source(wave):
    src = wave.sources
    V_f = wave.fluid_function_space

    # Vetor de fonte montado pelo próprio Spyro, com wavelet = 1
    rhs = fire.Cofunction(wave.function_space.dual())
    saved_wavelet, saved_current = src.wavelet, src.current_sources
    src.wavelet = np.ones_like(np.asarray(saved_wavelet))
    src.current_sources = [0]
    src.apply_source(rhs, 0)
    src.wavelet, src.current_sources = saved_wavelet, saved_current

    b_spyro = rhs.sub(0).dat.data_ro.reshape(-1).copy()

    X = fire.SpatialCoordinate(wave.submesh_fluid)
    u_z = fire.Function(V_f).interpolate(fire.as_vector([X[0], 0.0]))
    u_x = fire.Function(V_f).interpolate(fire.as_vector([0.0, X[1]]))

    a_z = float(np.dot(b_spyro, u_z.dat.data_ro.reshape(-1)))
    a_x = float(np.dot(b_spyro, u_x.dat.data_ro.reshape(-1)))

    u_zx = fire.Function(V_f).interpolate(fire.as_vector([X[1], 0.0]))  # u = (x, 0)
    u_xz = fire.Function(V_f).interpolate(fire.as_vector([0.0, X[0]]))  # u = (0, z)
    a_zx = float(np.dot(b_spyro, u_zx.dat.data_ro.reshape(-1)))
    a_xz = float(np.dot(b_spyro, u_xz.dat.data_ro.reshape(-1)))
    print(f"aplicada a u = (x, 0): {a_zx: .6e}   (fonte correta: 0)")
    print(f"aplicada a u = (0, z): {a_xz: .6e}   (fonte correta: 0)")

    print("=== Fonte de momento do Spyro (amplitude = I) ===")
    print(f"aplicada a u = (z, 0): {a_z: .6e}   (esperado: div u = 1 x escala)")
    print(f"aplicada a u = (0, x): {a_x: .6e}   (esperado: o MESMO valor)")
    if a_x != 0.0:
        print(f"razão z/x = {a_z / a_x: .6f}   (correto: 1.0)")
    return a_z, a_x
