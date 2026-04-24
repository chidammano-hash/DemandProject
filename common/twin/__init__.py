"""Digital Twin service — unified Monte Carlo state simulation.

Gen-4 Roadmap Cross-cutting #7. Today, three scripts rebuild inventory
state from SQL independently (``scripts/run_ss_simulation.py``,
``scripts/compute_inventory_projection.py``,
``scripts/generate_replenishment_exceptions.py``). This package centralizes
the state fetch and the Monte Carlo horizon simulation so consumers share
one abstraction.

Public API:
    - :class:`common.twin.state.TwinState` — per (item_id, loc) twin state

TODO(gen-4): Wire the remaining two consumers (inventory projection,
exception orchestrator) to ``TwinState.simulate`` and delete their
bespoke state loaders.
"""

from common.twin.state import TwinState

__all__ = ["TwinState"]
