"""Entry point for python -m adv_algorithm_testing.

Sets macOS fork-safety BEFORE any torch/neuralforecast imports to prevent
SIGSEGV in ProcessPoolExecutor children.
"""
import multiprocessing
import os
import sys

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
# OMP_NUM_THREADS=1: prevents OpenMP deadlock in PyTorch on macOS.
# When orphaned worker processes exist from prior runs, PyTorch's parallel
# tensor ops (__kmpc_fork_call) deadlock in __kmp_join_barrier waiting for
# threads that never respond.  Single-threaded OpenMP avoids this entirely.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

# Now safe to import and run
from adv_algorithm_testing.run_adv_expert_panel import main  # noqa: E402

main()
