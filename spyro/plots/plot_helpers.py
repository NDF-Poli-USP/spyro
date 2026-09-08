"""Helpers for use in other plotting methods."""

import matplotlib.pyplot as plt
from pathlib import Path


def _finalize_figure(
    fig,
    filename: str | Path | None = None,
    *,
    formats: tuple[str, ...] | None = None,
    show: bool = False,
    hold: bool = False,
    **savefig_kwargs,
):
    if filename is not None:
        filename = Path(filename)

        if formats is None:
            # Save exactly what the caller requested.
            fig.savefig(filename, **savefig_kwargs)
        else:
            # Replace or add extensions.
            stem = filename.with_suffix("")
            for fmt in formats:
                fig.savefig(
                    stem.with_suffix(f".{fmt}"),
                    **savefig_kwargs,
                )

    if show:
        plt.show()

    if not hold:
        plt.close(fig)

    return fig
