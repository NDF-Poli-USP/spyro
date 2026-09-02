"""Compara os resultados do AcousticElasticWave (Spyro) com o Gar6more2D.

USO:
  1) No script principal (fluid_solid_coupled_forward.py), depois de
     Wave_obj.forward_solve(), salve os dados uma vez:

       import numpy as np
       np.savez(
           "results/spyro_receiver_data.npz",
           p_spyro=np.asarray(Wave_obj.forward_solution_receivers)[:, 0],
           u_solid=np.array(Wave_obj.solid_receiver_history)[:, 0, :],
           dt=dictionary["time_axis"]["dt"],
           final_time=dictionary["time_axis"]["final_time"],
       )

  2) Depois, rode ESTE arquivo sozinho (sem precisar rodar a simulação de
     novo) quantas vezes quiser, pra ajustar sinal/cores/etc.:

       python3 compare_spyro_gar6more.py

Espera os arquivos do Gar6more2D em:
  results_fluid/P.dat        (height=+2.5, Source derivative=2)
  results_solid/Ux.dat,Uy.dat (height=-2.5, Source derivative=1)
"""
import os
import numpy as np

from spyro.plots.receiver_plots import plot_compare_receivers_array

GAR6MORE_DIR_FLUID = "/workspaces/spyro2/notebook_tutorials/results_fluid"
GAR6MORE_DIR_SOLID = "/workspaces/spyro2/notebook_tutorials/results_solid"


def load_gar6more_file(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        raise ValueError(f"{path}: só 1 coluna encontrada — esperado (tempo, valor).")
    return data[:, 0], data[:, 1]


def normalize(x):
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def _remove_pdfs(output_dir):
    for name in ("compare_P.pdf", "compare_ux.pdf", "compare_uz.pdf"):
        full = os.path.join(output_dir, name)
        if os.path.exists(full):
            os.remove(full)


def compare_with_gar6more_from_file(
    npz_path="results/spyro_receiver_data.npz",
    output_dir="results",
    flip_uz=False,
):
    """Lê dados já salvos do Spyro (npz) e compara com o Gar6more2D. Não
    depende de Wave_obj nem roda forward_solve() de novo — rápido, pode
    rodar quantas vezes quiser pra ajustar o gráfico.
    """
    data = np.load(npz_path)
    dt, final_time = float(data["dt"]), float(data["final_time"])
    nt = int(final_time / dt) + 1
    t_spyro = np.linspace(0.0, final_time, nt)

    p_spyro = data["p_spyro"]
    u_solid = data["u_solid"]
    uz_spyro = u_solid[:, 0]   # componente z (tangencial, mesh_z)
    ux_spyro = u_solid[:, 1]   # componente x (normal, mesh_x)

    # -------------------------------------------------------------
    t_gar6_p,  p_gar6      = load_gar6more_file(f"{GAR6MORE_DIR_FLUID}/P.dat")
    t_gar6_ux, ux_gar6_raw = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Ux.dat")  # -> u_z
    t_gar6_uy, uy_gar6_raw = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Uy.dat")  # -> u_x

    mask = t_spyro <= t_gar6_p[-1]
    t_spyro  = t_spyro[mask]
    p_spyro  = p_spyro[mask]
    uz_spyro = uz_spyro[mask]
    ux_spyro = ux_spyro[mask]

    p_gar6_i  = np.interp(t_spyro, t_gar6_p,  p_gar6,  left=0.0, right=0.0)
    uz_sign = -1.0 if flip_uz else 1.0
    uz_gar6_i = uz_sign * np.interp(t_spyro, t_gar6_ux, ux_gar6_raw, left=0.0, right=0.0)
    ux_gar6_i = -np.interp(t_spyro, t_gar6_uy, uy_gar6_raw, left=0.0, right=0.0)

    # -------------------------------------------------------------
    p_spyro_n,  p_gar6_n  = normalize(p_spyro),  normalize(p_gar6_i)
    uz_spyro_n, uz_gar6_n = normalize(uz_spyro), normalize(uz_gar6_i)
    ux_spyro_n, ux_gar6_n = normalize(ux_spyro), normalize(ux_gar6_i)

    # -------------------------------------------------------------
    plot_compare_receivers_array(
        receiver_data_first=p_spyro_n.reshape(-1, 1),
        receiver_data_second=p_gar6_n.reshape(-1, 1),
        time_values=t_spyro,
        first_label="Spyro",
        second_label="Gar6more2D",
        output_path=f"{output_dir}/compare_P",
    )
    plot_compare_receivers_array(
        receiver_data_first=ux_spyro_n.reshape(-1, 1),
        receiver_data_second=ux_gar6_n.reshape(-1, 1),
        time_values=t_spyro,
        first_label="Spyro",
        second_label="Gar6more2D",
        output_path=f"{output_dir}/compare_ux",
    )
    plot_compare_receivers_array(
        receiver_data_first=uz_spyro_n.reshape(-1, 1),
        receiver_data_second=uz_gar6_n.reshape(-1, 1),
        time_values=t_spyro,
        first_label="Spyro",
        second_label="Gar6more2D",
        output_path=f"{output_dir}/compare_uz",
    )

    _remove_pdfs(output_dir)
    print("Comparação salva: compare_P.png, compare_ux.png, compare_uz.png")


if __name__ == "__main__":
    # TROCA flip_uz=True AQUI pra testar o sinal invertido em u_z, sem
    # precisar mexer em mais nada — só rodar este arquivo de novo.
    compare_with_gar6more_from_file(flip_uz=False)