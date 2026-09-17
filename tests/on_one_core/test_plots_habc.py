import matplotlib

matplotlib.use("Agg")

from spyro.plots.plots_habc import create_folder, plot_function_layer_size


def test_create_folder_creates_directory(tmp_path):
    folder = tmp_path / "nested" / "output"

    create_folder(folder)

    assert folder.is_dir()


def test_plot_function_layer_size_saves_outputs(tmp_path):
    output_folder = tmp_path / "plots"

    plot_function_layer_size(
        layer_parameters=(0.5, 1.0),
        frequency_parameters=(10.0, 10.0),
        geometry_parameters=(0.25, 1.0),
        reference_frequency_layer_sizes=[0.1, 0.2, 0.3],
        output_folder=output_folder,
        show=False,
    )

    assert (output_folder / "layer_opts.png").exists()
    assert (output_folder / "layer_opts.pdf").exists()
