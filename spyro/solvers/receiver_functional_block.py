"""Custom pyadjoint blocks for receiver-space functionals."""

from __future__ import annotations

import numpy as np

import firedrake as fire
from pyadjoint import AdjFloat, Block
from pyadjoint.tape import annotate_tape, get_working_tape


def _as_receiver_array(values):
    if isinstance(values, fire.Function):
        return np.asarray(values.dat.data_ro, dtype=float)
    return np.asarray(values, dtype=float)


class ReceiverL2FunctionalBlock(Block):
    """L2 receiver misfit contribution as one pyadjoint block.

    This fuses two operations that are otherwise represented on the tape as a
    Firedrake scalar assembly plus the derivative machinery around that
    assembly.  The dependency is the predicted receiver trace at one time step,
    represented as a DG0 Function on a VertexOnlyMesh.  The observed receiver
    values are fixed data for the reduced functional evaluation.
    """

    def __init__(self, predicted_receivers, observed_receivers, scale):
        super().__init__()
        self.add_dependency(predicted_receivers)
        self.observed_receivers = np.array(
            _as_receiver_array(observed_receivers),
            dtype=float,
            copy=True,
        )
        self.scale = float(scale)

    def _residual(self, predicted_receivers):
        return self.observed_receivers - _as_receiver_array(predicted_receivers)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        predicted_receivers, = inputs
        residual = self._residual(predicted_receivers)
        return AdjFloat(self.scale * float(np.dot(residual, residual)))

    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        predicted_receivers, = inputs
        adj_input, = adj_inputs
        residual = self._residual(predicted_receivers)
        derivative = -2.0 * self.scale * float(adj_input) * residual

        adjoint = fire.Cofunction(predicted_receivers.function_space().dual())
        adjoint.dat.data[:] = derivative
        return adjoint

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        predicted_receivers, = inputs
        predicted_tlm, = tlm_inputs
        if predicted_tlm is None:
            return AdjFloat(0.0)

        residual = self._residual(predicted_receivers)
        tlm_values = _as_receiver_array(predicted_tlm)
        return AdjFloat(-2.0 * self.scale * float(np.dot(residual, tlm_values)))


def receiver_l2_functional(predicted_receivers, observed_receivers, scale):
    """Return one receiver L2 contribution, optionally annotated on the tape."""
    residual = _as_receiver_array(observed_receivers) - _as_receiver_array(
        predicted_receivers
    )
    value = AdjFloat(float(scale) * float(np.dot(residual, residual)))

    if annotate_tape():
        block = ReceiverL2FunctionalBlock(
            predicted_receivers,
            observed_receivers,
            scale,
        )
        get_working_tape().add_block(block)
        block.add_output(value.block_variable)

    return value


def ensure_receiver_maps(receivers):
    """Build classic point-interpolation maps when VOM skipped them."""
    if receivers.cellNodeMaps is None or receivers.cell_tabulations is None:
        if receivers.is_local is None:
            receivers.is_local = [0] * receivers.number_of_points
        receivers.build_maps()


def receiver_values_from_field(field, receivers):
    """Interpolate a scalar field to receiver points with Spyro's maps."""
    ensure_receiver_maps(receivers)
    return np.asarray(
        receivers.interpolate(field.dat.data_ro_with_halos[:]),
        dtype=float,
    )


class FieldReceiverL2FunctionalBlock(Block):
    """Receiver L2 contribution depending directly on the wave field.

    The block represents

        scale * ||d_obs - R u||_2^2

    where ``R`` is the point receiver interpolation operator represented by
    Spyro's precomputed cell-node maps and basis tabulations.
    """

    def __init__(self, field, receivers, observed_receivers, scale):
        super().__init__()
        self.add_dependency(field)
        self.receivers = receivers
        ensure_receiver_maps(receivers)
        self.receiver_maps = []
        for receiver_id in range(receivers.number_of_points):
            if receivers.is_local[receiver_id] is not None:
                self.receiver_maps.append(
                    (
                        np.asarray(receivers.cellNodeMaps[receiver_id], dtype=int),
                        np.asarray(
                            receivers.cell_tabulations[receiver_id],
                            dtype=float,
                        ).reshape(-1),
                    )
                )
            else:
                self.receiver_maps.append(None)
        self.observed_receivers = np.array(
            _as_receiver_array(observed_receivers),
            dtype=float,
            copy=True,
        )
        self.scale = float(scale)
        self._last_residual = self._residual(field)

    def _receiver_values(self, field):
        return receiver_values_from_field(field, self.receivers)

    def _residual(self, field):
        return self.observed_receivers - self._receiver_values(field)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        field, = inputs
        residual = self._residual(field)
        self._last_residual = residual
        return AdjFloat(self.scale * float(np.dot(residual, residual)))

    def evaluate_adj_component(
        self, inputs, adj_inputs, block_variable, idx, prepared=None
    ):
        field, = inputs
        adj_input, = adj_inputs
        residual = self._last_residual
        if residual is None:
            residual = self._residual(field)
        receiver_adjoint = -2.0 * self.scale * float(adj_input) * residual

        adjoint = fire.Cofunction(field.function_space().dual())
        adjoint.dat.data_with_halos[:] = 0.0
        for receiver_map, value in zip(self.receiver_maps, receiver_adjoint):
            if receiver_map is not None:
                node_ids, phis = receiver_map
                adjoint.dat.data_with_halos[node_ids] += value * phis
        return adjoint

    def evaluate_tlm_component(
        self, inputs, tlm_inputs, block_variable, idx, prepared=None
    ):
        field, = inputs
        field_tlm, = tlm_inputs
        if field_tlm is None:
            return AdjFloat(0.0)

        residual = self._residual(field)
        tlm_values = self._receiver_values(field_tlm)
        return AdjFloat(-2.0 * self.scale * float(np.dot(residual, tlm_values)))


def field_receiver_l2_functional(field, receivers, observed_receivers, scale):
    """Return a receiver L2 contribution annotated against the wave field."""
    residual = _as_receiver_array(observed_receivers) - receiver_values_from_field(
        field,
        receivers,
    )
    value = AdjFloat(float(scale) * float(np.dot(residual, residual)))

    if annotate_tape():
        block = FieldReceiverL2FunctionalBlock(
            field,
            receivers,
            observed_receivers,
            scale,
        )
        get_working_tape().add_block(block)
        block.add_output(value.block_variable)

    return value
