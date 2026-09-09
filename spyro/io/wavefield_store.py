"""Contiguous storage for the forward wavefield used by the implemented adjoint.

Why this exists
---------------
The implemented adjoint stores one ``fire.Function`` per sampled time step.
That costs three separate things:

1. Retention. Firedrake's ``Function.assign`` places the result in a reference
   cycle (Function / CoordinatelessFunction / Dat / Vec), so refcounting never
   frees it; only the cyclic collector can, and that collector triggers on
   Python object counts. A Function is a small Python object wrapping a large
   PETSc buffer, so it effectively never fires under this workload. Measured:
   one full wavefield retained per gradient, growing without bound.
2. Per-object overhead. Measured at ~22.6 kB per snapshot on top of the data
   (PETSc Vec, PyOP2 Dat, UFL Coefficient, layout metadata). That is 5% at
   nt=1000 and larger than the data itself past ~20k steps.
3. A hard float64 floor, because the storage *is* the Function.

Storing into preallocated numpy blocks removes all three: no Function is
created inside the loop, so there is nothing to retain; the per-object cost is
amortised over ``block_size`` snapshots; and the dtype becomes a free choice.

Reference numbers (55401 dofs, nt=1000, serial): 423 MB of data, 445 MB
measured with the Function-per-step scheme, 211 MB if stored as float32.

Storage layout
--------------
Blocks rather than one big array, and rather than a doubling vector. A doubling
vector reallocates and transiently holds ~1.5x the wavefield, which is exactly
the peak this class is meant to lower. Fixed blocks never realloc, and they
preserve the existing behaviour of not reserving the whole nt-step wavefield up
front -- a solve that aborts early (the instability check in the forward loop)
still only pays for what it actually stored.

Halos are kept. The full ``data_with_halos`` range is copied in and out, which
reproduces the current semantics exactly and needs no communication. Storing
only owned dofs would save a further 10-30% in parallel but requires a halo
update on read-back; that is deliberately left for a separate change so this
one stays numerically identical at float64.

Precision
---------
``dtype=np.float64`` is bit-for-bit identical to the Function-per-step scheme:
the same doubles, in a different container. ``np.float32`` halves the footprint
and is standard practice for stored forward wavefields in FWI, but it does
change the gradient and must be validated against the finite-difference tests
before being trusted.
"""

from __future__ import annotations

import numpy as np


class WavefieldStore:
    """Append-and-pop storage for forward snapshots, backed by numpy blocks.

    Supports the access pattern the adjoint actually uses -- append during the
    forward sweep, pop from the end during the backward sweep, with read access
    to the two samples behind the one being popped -- and nothing else. It is
    deliberately not a drop-in ``list``: an incomplete sequence emulation would
    silently do the wrong thing in the places that iterate over snapshots.

    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Space the snapshots live in. Used once, to size a row.
    dtype : numpy dtype, optional
        Storage precision. float64 (default) reproduces the previous behaviour
        exactly; float32 halves memory and changes the gradient.
    block_size : int, optional
        Snapshots per numpy block.
    """

    def __init__(self, function_space, dtype=np.float64, block_size: int = 64):
        import firedrake as fire

        probe = fire.Function(function_space)
        self._row_shape = probe.dat.data_with_halos.shape
        self._row_size = int(np.prod(self._row_shape))
        del probe

        self._function_space = function_space
        self._dtype = np.dtype(dtype)
        self._block_size = int(block_size)
        self._blocks: list[np.ndarray] = []
        self._n = 0

    # ------------------------------------------------------------------ size

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        # Several call sites test `if wave.forward_solution:` to mean "is there
        # a stored wavefield". Without this, an empty store would be truthy
        # (default object truthiness) and those branches would invert.
        return self._n > 0

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self._blocks)

    @property
    def dtype(self):
        return self._dtype

    # ----------------------------------------------------------- write side

    def _row(self, index: int) -> np.ndarray:
        block, offset = divmod(index, self._block_size)
        return self._blocks[block][offset]

    def append(self, function) -> None:
        """Copy a Function's dofs (including halos) into the next slot."""
        if self._n % self._block_size == 0:
            self._blocks.append(
                np.empty((self._block_size, self._row_size), dtype=self._dtype)
            )
        self._row(self._n)[:] = function.dat.data_ro_with_halos.reshape(-1)
        self._n += 1

    # ------------------------------------------------------------ read side

    def peek_into(self, out, offset_from_end: int = 0) -> None:
        """Write the sample ``offset_from_end`` back from the end into ``out``."""
        index = self._n - 1 - offset_from_end
        if index < 0:
            raise IndexError("wavefield store exhausted")
        out.dat.data_with_halos[:] = self._row(index).reshape(self._row_shape)

    def pop_into(self, out) -> None:
        """Write the last sample into ``out`` and drop it from the store."""
        self.peek_into(out, 0)
        self._drop_last()

    def pop_second_derivative_into(self, out, sample_dt2: float) -> None:
        """Second time-derivative of the stored field, into ``out``.

        Consumes one sample, matching ``_compute_dufordt2``: a 3-point centred
        stencil over the last three samples when available, and a one-sided
        fallback near the start of the forward record, where the field is ~0.

        The arithmetic is done on the numpy rows rather than as a UFL
        expression, which also removes the per-step Constant and the per-step
        expression assembly.
        """
        if self._n > 2:
            u_kp1 = self._row(self._n - 1)
            u_k = self._row(self._n - 2)
            u_km1 = self._row(self._n - 3)
            value = (u_kp1 - 2.0 * u_k + u_km1) / sample_dt2
        elif self._n > 0:
            value = self._row(self._n - 1) / sample_dt2
        else:
            raise IndexError("wavefield store exhausted")

        out.dat.data_with_halos[:] = value.reshape(self._row_shape)
        self._drop_last()

    def _drop_last(self) -> None:
        self._n -= 1
        # Release whole blocks as the backward sweep consumes them, so the
        # footprint falls during the adjoint instead of only at the end.
        while len(self._blocks) * self._block_size >= self._n + self._block_size:
            if not self._blocks:
                break
            self._blocks.pop()

    # ----------------------------------------------------- (de)serialisation

    def to_arrays(self) -> list[np.ndarray]:
        """Snapshots as a list of 1-D arrays, for saving to disk."""
        return [self._row(i).copy() for i in range(self._n)]

    def extend_from_arrays(self, arrays) -> None:
        """Refill the store from previously saved rows."""
        for array in arrays:
            if self._n % self._block_size == 0:
                self._blocks.append(
                    np.empty(
                        (self._block_size, self._row_size), dtype=self._dtype
                    )
                )
            self._row(self._n)[:] = np.asarray(array).reshape(-1)
            self._n += 1

    def clear(self) -> None:
        self._blocks.clear()
        self._n = 0
