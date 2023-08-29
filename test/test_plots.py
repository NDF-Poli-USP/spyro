import spyro


def test_plot():
    Wave_obj = spyro.examples.Rectangle_acoustic()
    Wave_obj.forward_solve()
    spyro.plots.plot_shots(Wave_obj, show=True)


if __name__ == "__main__":
    test_plot()
