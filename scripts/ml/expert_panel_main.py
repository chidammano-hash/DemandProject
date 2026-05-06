"""Entry point for ``python -m common.ml.expert_panel``.

Sets macOS fork-safety BEFORE any torch/neuralforecast imports to prevent
SIGSEGV in ProcessPoolExecutor children.

Defaults to the advanced expert panel, since that is the path that needs
the fork-safety setup. The base panel is invoked via the fully-qualified
module path ``python -m common.ml.expert_panel.run_expert_panel``.
"""
import multiprocessing
import os
import sys

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
# OMP_NUM_THREADS=1: prevents OpenMP deadlock in PyTorch on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

# Now safe to import and run
from scripts.ml.run_adv_expert_panel import main  # noqa: E402

main()
