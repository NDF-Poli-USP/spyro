from spyro.model.time_axis import TimeAxis


class TestTimeAxis:

    def test_constructor(self):
        t = TimeAxis(initial_time=1.0, final_time=5.0, dt=1.0)

        assert t.initial_time == 1.0
        assert t.final_time == 5.0
        assert t.dt == 1.0

    def test_update(self):
        t = TimeAxis(initial_time=1.0, final_time=5.0, dt=1.0)

        t.update()

        assert t.current_time == 2.0
        assert t.index == 1

    def test_get_reverse_index(self):
        t = TimeAxis(initial_time=1.0, final_time=5.0, dt=1.0)

        assert t.get_reverse_index() == 4

    def test_can_update(self):
        t = TimeAxis(initial_time=1.0, final_time=2.0, dt=1.0)

        assert t.can_update()
        t.update()
        assert not t.can_update()

    def test_copy(self):
        t = TimeAxis(initial_time=1.0, final_time=5.0, dt=1.0)

        t_copy = t.copy()

        assert t.initial_time == t_copy.initial_time
        assert t.final_time == t_copy.final_time
