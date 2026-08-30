# If JAX was already imported in this notebook before running this cell,
# restart the kernel first so the env vars below actually take effect.

import os
import gc
import json
import csv
import time
import socket
import faulthandler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

faulthandler.enable(all_threads=True)

# ============================================================
# IMPORTANT: set CPU-device count BEFORE importing JAX
# ============================================================
NUM_CPU_DEVICES = min(1, os.cpu_count() or 1)
os.environ.setdefault("JAX_NUM_CPU_DEVICES", str(NUM_CPU_DEVICES))

import numpy as np
import joblib
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

from tslearn.datasets import UCR_UEA_datasets

import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

try:
    from jax.experimental.compilation_cache import compilation_cache
    compilation_cache.set_cache_dir("/tmp/jax_compilation_cache")
except Exception:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("JAX local_device_count():", jax.local_device_count())

# ============================================================
# Local imports
# ============================================================

import Salvi as SIG

from tensordev.sss.kernel import FSSK
from tensordev.kernel.fssk import FSSKSigKernel
from tensordev.kernel.static_kernels import RBFKernel

# ============================================================
# User settings
# ============================================================

_datasets = [
    "BasicMotions",
]

# Only time augmentation, no lead-lag
TRANSFORMS = [(True, False)]

# Scaling choices optimized inside Optuna:
# "global" = divide by max(abs(x_train))
# "std"    = divide by std(x_train)
SCALING_METHODS = ["std"]

# We optimize R and scaling_kind inside Optuna.
# The outer loop is only over the static kernel kind.
METHOD_SPECS = [
    #{"static_kernel_kind": "linear"},
    {"static_kernel_kind": "rbf"},
]

VSIG_BACKEND = "scan"
VSIG_SCHEME = "heun"
VSIG_OUTSIDE_WARMUP = True
VSIG_NUM_DEVICES = max(1, min(NUM_CPU_DEVICES, jax.local_device_count()))

CLEAR_CACHES_EVERY = 20

OPTUNA_SEED = 1234
OPTUNA_N_TRIALS = 200
OPTUNA_STARTUP_TRIALS = min(10, OPTUNA_N_TRIALS)

OPTUNA_C_MIN = 1.0
OPTUNA_C_MAX = 1e4
OPTUNA_DYADIC_CHOICES = [0]

# R is optimized inside Optuna
OPTUNA_STATE_RANK_CHOICES = [2]

# R = 1 search space
OPTUNA_R1_LAMBDA_MIN = 1e-2
OPTUNA_R1_LAMBDA_MAX = 4.0
OPTUNA_R1_ALPHA_MIN = 5e-4
OPTUNA_R1_ALPHA_MAX = 1.0

# ============================================================
# R = 2 free parameter search space
# ============================================================

OPTUNA_R2_LAMBDA1_MIN = 1e-2
OPTUNA_R2_LAMBDA1_MAX = 4.0

OPTUNA_R2_LAMBDA2_MIN = 1e-2
OPTUNA_R2_LAMBDA2_MAX = 4.0

OPTUNA_R2_ALPHA1_MIN = 5e-4
OPTUNA_R2_ALPHA1_MAX = 4.0

OPTUNA_R2_ALPHA2_MIN = 5e-4
OPTUNA_R2_ALPHA2_MAX = 4.0

# Free skew coupling in
# Lambda = [[lambda1,  coupling],
#           [-coupling, lambda2]]
OPTUNA_R2_COUPLING_MIN = 0
OPTUNA_R2_COUPLING_MAX = 2.0

# RBF sigma grid
OPTUNA_RBF_SIGMA_GRID = [
    1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2,
    1e-1, 2.5e-1, 5e-1, 7.5e-1, 1.0, 2.0, 5.0, 10.0
]

# Safety threshold: prune exploding Gram matrices
GRAM_MAX_ABS = 1e8

# Saving
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "volterra_optuna_runs_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TAG = f"{socket.gethostname()}__pid{os.getpid()}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MERGE_SUMMARIES_AT_END = False


# ============================================================
# Small helpers
# ============================================================

def maybe_cleanup(step_idx=None, force=False):
    if force or (
        step_idx is not None
        and CLEAR_CACHES_EVERY > 0
        and (step_idx + 1) % CLEAR_CACHES_EVERY == 0
    ):
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()
        try:
            print(f"[cleanup] live arrays: {len(jax.live_arrays())}")
        except Exception:
            pass


def make_uniform_grid(length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, length, dtype=dtype, device=device)


def choose_device_and_dtype_volterra(_x_np):
    return torch.device("cpu"), torch.float64


def scale_train_test(x_train, x_test=None, scaling_kind: str = "std", eps: float = 1e-12):
    """
    Train-only scaling.

    scaling_kind:
        "global" : divide by max(abs(x_train))
        "std"    : divide by std(x_train)

    No mean subtraction.
    """
    x_train = np.asarray(x_train, dtype=np.float64)

    if scaling_kind == "global":
        scale = np.max(np.abs(x_train))

    elif scaling_kind == "std":
        scale = np.std(x_train)

    else:
        raise ValueError(
            f"Unknown scaling_kind={scaling_kind!r}. "
            "Use 'global' or 'std'."
        )

    if not np.isfinite(scale) or scale < eps:
        scale = 1.0

    x_train_scaled = x_train / scale

    if x_test is None:
        return x_train_scaled

    x_test = np.asarray(x_test, dtype=np.float64)
    x_test_scaled = x_test / scale

    return x_train_scaled, x_test_scaled


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_jax(x):
    if isinstance(x, jnp.ndarray):
        return x
    if hasattr(x, "detach"):
        return jnp.asarray(x.detach().cpu().numpy())
    return jnp.asarray(x)


def sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


def method_key(static_kernel_kind: str) -> str:
    return static_kernel_kind


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def study_to_records(study: optuna.Study) -> List[Dict[str, Any]]:
    rows = []
    for tr in study.trials:
        rows.append(
            {
                "number": tr.number,
                "state": str(tr.state),
                "value": None if tr.value is None else float(tr.value),
                "params": to_builtin(tr.params),
                "user_attrs": to_builtin(tr.user_attrs),
            }
        )
    return rows


def save_completed_result(bundle: Dict[str, Any]) -> None:
    dataset = sanitize(bundle["dataset"])
    mkey = sanitize(bundle["method_key"])
    prefix = f"{dataset}__{mkey}__{RUN_TAG}"

    model_path = OUTPUT_DIR / f"{prefix}.joblib"
    summary_path = OUTPUT_DIR / f"{prefix}.summary.json"

    joblib.dump(bundle, model_path, compress=3)

    summary = {
        "dataset": bundle["dataset"],
        "method_key": bundle["method_key"],
        "scaling_kind": bundle["scaling_kind"],
        "state_rank": bundle["state_rank"],
        "static_kernel_kind": bundle["static_kernel_kind"],
        "subsample": bundle["subsample"],
        "transform": bundle["transform"],
        "train_shape": bundle["train_shape"],
        "test_shape": bundle["test_shape"],
        "best_params": to_builtin(bundle["best_params"]),
        "resolved_params": to_builtin(bundle["resolved_params"]),
        "train_cv_score": bundle["train_cv_score"],
        "test_score": bundle["test_score"],
        "refit_C": bundle["refit_C"],
        "timings": to_builtin(bundle["timings"]),
        "run_tag": RUN_TAG,
        "hostname": socket.gethostname(),
        "model_path": str(model_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {summary_path.name}")
    print(f"[saved] {model_path.name}")


def merge_saved_summaries(output_dir: Path) -> None:
    rows = []
    for p in sorted(output_dir.glob("*.summary.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"[merge] skipping {p.name}: {e}")

    merged_json = output_dir / "merged_summary.json"
    merged_csv = output_dir / "merged_summary.csv"

    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(merged_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()
                    }
                )

    print(f"[merge] wrote {merged_json}")
    print(f"[merge] wrote {merged_csv}")


def kernel_health_stats(K: np.ndarray) -> Dict[str, float]:
    K = np.asarray(K)
    diag = np.diag(K) if K.ndim == 2 and K.shape[0] == K.shape[1] else np.array([])
    out = {
        "min": float(np.nanmin(K)),
        "max": float(np.nanmax(K)),
        "max_abs": float(np.nanmax(np.abs(K))),
    }
    if diag.size > 0:
        out["diag_min"] = float(np.nanmin(diag))
        out["diag_max"] = float(np.nanmax(diag))
    return out


def gram_is_usable(K: np.ndarray, max_abs: float = GRAM_MAX_ABS) -> bool:
    if not np.all(np.isfinite(K)):
        return False
    if np.max(np.abs(K)) > max_abs:
        return False
    return True


# ============================================================
# Parameter builders
# ============================================================

def build_method_matrices_from_torch(
    x_ref_torch: torch.Tensor,
    state_rank: int,
    dyadic_order: int,

    # R=1 parameters
    lambda_base: Optional[float] = None,
    alpha_scale: Optional[float] = None,

    # R=2 free parameters
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    coupling: Optional[float] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
):
    device = x_ref_torch.device
    dtype = x_ref_torch.dtype
    _, length, d = x_ref_torch.shape

    q = 1
    m = d

    s_grid = make_uniform_grid(length, dtype=dtype, device=device)
    t_grid = make_uniform_grid(length, dtype=dtype, device=device)

    A = torch.zeros((q, m, d), dtype=dtype, device=device)
    A[0] = torch.eye(d, dtype=dtype, device=device)

    if state_rank == 1:
        if lambda_base is None or alpha_scale is None:
            raise ValueError("R=1 requires lambda_base and alpha_scale.")

        Lambda = torch.tensor(
            [[float(lambda_base)]],
            dtype=dtype,
            device=device,
        )
        b = torch.tensor(
            [[float(alpha_scale)]],
            dtype=dtype,
            device=device,
        )

        resolved = {
            "state_rank": 1,
            "lambda1": float(lambda_base),
            "lambda2": None,
            "coupling": 0.0,
            "alpha1": float(alpha_scale),
            "alpha2": None,
            "dyadic_order": int(dyadic_order),
        }

    elif state_rank == 2:
        if (
            lambda1 is None
            or lambda2 is None
            or coupling is None
            or alpha1 is None
            or alpha2 is None
        ):
            raise ValueError("R=2 requires lambda1, lambda2, coupling, alpha1, alpha2.")

        lambda1 = float(lambda1)
        lambda2 = float(lambda2)
        coupling = float(coupling)
        alpha1 = float(alpha1)
        alpha2 = float(alpha2)

        Lambda = torch.tensor(
            [
                [lambda1, coupling],
                [-coupling, lambda2],
            ],
            dtype=dtype,
            device=device,
        )

        b = torch.tensor(
            [[alpha1, alpha2]],
            dtype=dtype,
            device=device,
        )

        resolved = {
            "state_rank": 2,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "coupling": coupling,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "dyadic_order": int(dyadic_order),
        }

    else:
        raise ValueError(f"Unsupported state_rank={state_rank}")

    dt_x0 = float((s_grid[1] - s_grid[0]).item())
    dt_y0 = float((t_grid[1] - t_grid[0]).item())

    return Lambda, A, b, dt_x0, dt_y0, resolved


# ============================================================
# VSIG JAX backend
# ============================================================

def build_vsig_kernel_from_torch(
    x_ref_torch: torch.Tensor,
    state_rank: int,
    static_kernel_kind: str,
    dyadic_order: int,
    sigma: Optional[float] = None,

    # R=1
    lambda_base: Optional[float] = None,
    alpha_scale: Optional[float] = None,

    # R=2 free
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    coupling: Optional[float] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
):
    Lambda, A, b, dt_x0, dt_y0, resolved = build_method_matrices_from_torch(
        x_ref_torch=x_ref_torch,
        state_rank=state_rank,
        dyadic_order=dyadic_order,
        lambda_base=lambda_base,
        alpha_scale=alpha_scale,
        lambda1=lambda1,
        lambda2=lambda2,
        coupling=coupling,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    ker_coupled = FSSK.from_matrix(
        Lambda=to_jax(Lambda),
        A=to_jax(A),
        b=to_jax(b),
    )

    kwargs = dict(
        kernel=ker_coupled,
        dt_x=dt_x0,
        dt_y=dt_y0,
        backend=VSIG_BACKEND,
        dyadic_order=dyadic_order,
        scheme=VSIG_SCHEME,
        precompute_propagators=True,
        num_devices=VSIG_NUM_DEVICES,
    )

    if static_kernel_kind == "linear":
        pass

    elif static_kernel_kind == "rbf":
        if sigma is None:
            raise ValueError("sigma must be provided for static_kernel_kind='rbf'.")
        kwargs["static_kernel"] = RBFKernel(sigma=float(sigma))
        resolved["sigma"] = float(sigma)

    else:
        raise ValueError(f"Unknown static_kernel_kind={static_kernel_kind!r}")

    sigker_vsig = FSSKSigKernel(**kwargs)
    return sigker_vsig, resolved


def warmup_vsig_once_for_dataset_shape(
    x_train_torch: torch.Tensor,
    state_rank: int,
    static_kernel_kind: str,
):
    warmup_kwargs = dict(
        x_ref_torch=x_train_torch,
        state_rank=state_rank,
        static_kernel_kind=static_kernel_kind,
        dyadic_order=OPTUNA_DYADIC_CHOICES[0],
    )

    if state_rank == 1:
        warmup_kwargs.update(
            lambda_base=1.0,
            alpha_scale=1.0,
            sigma=(1.0 if static_kernel_kind == "rbf" else None),
        )

    else:
        warmup_kwargs.update(
            lambda1=0.5,
            lambda2=2.0,
            coupling=0.0,
            alpha1=0.5,
            alpha2=0.5,
            sigma=(1.0 if static_kernel_kind == "rbf" else None),
        )

    sigker_vsig, _ = build_vsig_kernel_from_torch(**warmup_kwargs)

    X_jax = to_jax(x_train_torch)
    G = sigker_vsig.compute_Gram(X_jax, X_jax)
    _ = jax.block_until_ready(G)

    del sigker_vsig, X_jax, G
    gc.collect()


def warmup_vsig_all_ranks_for_dataset_shape(
    x_train_torch: torch.Tensor,
    static_kernel_kind: str,
):
    warmup_vsig_once_for_dataset_shape(
        x_train_torch=x_train_torch,
        state_rank=1,
        static_kernel_kind=static_kernel_kind,
    )
    warmup_vsig_once_for_dataset_shape(
        x_train_torch=x_train_torch,
        state_rank=2,
        static_kernel_kind=static_kernel_kind,
    )


def compute_vsig_train_gram(
    x_train_torch: torch.Tensor,
    state_rank: int,
    static_kernel_kind: str,
    dyadic_order: int,
    sigma: Optional[float] = None,

    # R=1
    lambda_base: Optional[float] = None,
    alpha_scale: Optional[float] = None,

    # R=2 free
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    coupling: Optional[float] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
):
    sigker_vsig, resolved = build_vsig_kernel_from_torch(
        x_ref_torch=x_train_torch,
        state_rank=state_rank,
        static_kernel_kind=static_kernel_kind,
        dyadic_order=dyadic_order,
        sigma=sigma,
        lambda_base=lambda_base,
        alpha_scale=alpha_scale,
        lambda1=lambda1,
        lambda2=lambda2,
        coupling=coupling,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    X_jax = to_jax(x_train_torch)
    G_train = sigker_vsig.compute_Gram(X_jax, X_jax)
    G_train = np.asarray(jax.device_get(jax.block_until_ready(G_train)))

    del sigker_vsig, X_jax
    gc.collect()

    return G_train, resolved


def compute_vsig_test_gram(
    x_train_torch: torch.Tensor,
    x_test_torch: torch.Tensor,
    state_rank: int,
    static_kernel_kind: str,
    dyadic_order: int,
    sigma: Optional[float] = None,

    # R=1
    lambda_base: Optional[float] = None,
    alpha_scale: Optional[float] = None,

    # R=2 free
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    coupling: Optional[float] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
):
    sigker_vsig, resolved = build_vsig_kernel_from_torch(
        x_ref_torch=x_train_torch,
        state_rank=state_rank,
        static_kernel_kind=static_kernel_kind,
        dyadic_order=dyadic_order,
        sigma=sigma,
        lambda_base=lambda_base,
        alpha_scale=alpha_scale,
        lambda1=lambda1,
        lambda2=lambda2,
        coupling=coupling,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    X_train_jax = to_jax(x_train_torch)
    X_test_jax = to_jax(x_test_torch)

    G_test = sigker_vsig.compute_Gram(X_test_jax, X_train_jax)
    G_test = np.asarray(jax.device_get(jax.block_until_ready(G_test)))

    del sigker_vsig, X_train_jax, X_test_jax
    gc.collect()

    return G_test, resolved


# ============================================================
# Optuna objective
# ============================================================

def make_volterra_objective(
    x_train_torch_by_scaling: Dict[str, torch.Tensor],
    y_train: np.ndarray,
    static_kernel_kind: str,
):
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=OPTUNA_SEED)

    def objective(trial: optuna.trial.Trial) -> float:
        scaling_kind = trial.suggest_categorical("scaling_kind", SCALING_METHODS)
        x_train_torch_vol = x_train_torch_by_scaling[scaling_kind]

        state_rank = trial.suggest_categorical("state_rank", OPTUNA_STATE_RANK_CHOICES)

        lambda_base = None
        alpha_scale = None

        lambda1 = None
        lambda2 = None
        coupling = None
        alpha1 = None
        alpha2 = None

        if state_rank == 1:
            lambda_base = trial.suggest_float(
                "lambda_base",
                OPTUNA_R1_LAMBDA_MIN,
                OPTUNA_R1_LAMBDA_MAX,
                log=True,
            )
            alpha_scale = trial.suggest_float(
                "alpha_scale",
                OPTUNA_R1_ALPHA_MIN,
                OPTUNA_R1_ALPHA_MAX,
                log=True,
            )

        elif state_rank == 2:
            lambda1 = trial.suggest_float(
                "lambda1",
                OPTUNA_R2_LAMBDA1_MIN,
                OPTUNA_R2_LAMBDA1_MAX,
                log=True,
            )
            lambda2 = trial.suggest_float(
                "lambda2",
                OPTUNA_R2_LAMBDA2_MIN,
                OPTUNA_R2_LAMBDA2_MAX,
                log=True,
            )
            coupling = trial.suggest_float(
                "coupling",
                OPTUNA_R2_COUPLING_MIN,
                OPTUNA_R2_COUPLING_MAX,
            )
            alpha1 = trial.suggest_float(
                "alpha1",
                OPTUNA_R2_ALPHA1_MIN,
                OPTUNA_R2_ALPHA1_MAX,
                log=True,
            )
            alpha2 = trial.suggest_float(
                "alpha2",
                OPTUNA_R2_ALPHA2_MIN,
                OPTUNA_R2_ALPHA2_MAX,
                log=True,
            )

        else:
            raise ValueError(f"Unsupported state_rank={state_rank}")

        dyadic_order = trial.suggest_categorical("dyadic_order", OPTUNA_DYADIC_CHOICES)
        C = trial.suggest_float("C", OPTUNA_C_MIN, OPTUNA_C_MAX, log=True)

        sigma = None
        if static_kernel_kind == "rbf":
            sigma = trial.suggest_categorical("sigma", OPTUNA_RBF_SIGMA_GRID)

        t0 = time.perf_counter()

        with torch.no_grad():
            G_train, resolved = compute_vsig_train_gram(
                x_train_torch=x_train_torch_vol,
                state_rank=state_rank,
                static_kernel_kind=static_kernel_kind,
                dyadic_order=dyadic_order,
                sigma=sigma,
                lambda_base=lambda_base,
                alpha_scale=alpha_scale,
                lambda1=lambda1,
                lambda2=lambda2,
                coupling=coupling,
                alpha1=alpha1,
                alpha2=alpha2,
            )

        gram_time = time.perf_counter() - t0

        resolved["scaling_kind"] = scaling_kind

        raw_stats = kernel_health_stats(G_train)
        trial.set_user_attr("raw_kernel_stats", to_builtin(raw_stats))
        trial.set_user_attr("resolved_params", to_builtin(resolved))

        if not gram_is_usable(G_train, max_abs=GRAM_MAX_ABS):
            print(
                f"[prune] bad Gram matrix | "
                f"scaling={scaling_kind}, "
                f"R={state_rank}, static={static_kernel_kind}, "
                f"lambda_base={lambda_base}, alpha_scale={alpha_scale}, "
                f"lambda1={lambda1}, lambda2={lambda2}, coupling={coupling}, "
                f"alpha1={alpha1}, alpha2={alpha2}, sigma={sigma}, "
                f"dyadic_order={dyadic_order}, C={C} | stats={raw_stats}"
            )
            raise optuna.TrialPruned()

        t1 = time.perf_counter()

        svc = SVC(
            C=C,
            kernel="precomputed",
            decision_function_shape="ovo",
        )

        try:
            scores = cross_val_score(
                svc,
                G_train,
                y_train,
                cv=cv_splitter,
                n_jobs=1,
                error_score="raise",
            )

        except Exception as e:
            print(
                f"[prune] SVM failed: {repr(e)} | "
                f"scaling={scaling_kind}, "
                f"R={state_rank}, static={static_kernel_kind}, "
                f"lambda_base={lambda_base}, alpha_scale={alpha_scale}, "
                f"lambda1={lambda1}, lambda2={lambda2}, coupling={coupling}, "
                f"alpha1={alpha1}, alpha2={alpha2}, sigma={sigma}, "
                f"dyadic_order={dyadic_order}, C={C}"
            )
            print(f"[prune] raw kernel stats: {raw_stats}")
            raise optuna.TrialPruned()

        fit_time = time.perf_counter() - t1
        score = float(np.mean(scores))

        trial.set_user_attr("gram_time", gram_time)
        trial.set_user_attr("fit_time", fit_time)

        if state_rank == 1:
            print(
                f"[trial {trial.number:03d}] "
                f"kernel={static_kernel_kind:>6s} "
                f"scale={scaling_kind:>6s} "
                f"R=1 σ={sigma if sigma is not None else '-'} "
                f"λ={lambda_base:.4g} α={alpha_scale:.4g} "
                f"dy={dyadic_order} C={C:.4g} "
                f"| gram={gram_time:7.3f}s cv={fit_time:7.3f}s "
                f"| score={score:.4f}"
            )

        else:
            print(
                f"[trial {trial.number:03d}] "
                f"kernel={static_kernel_kind:>6s} "
                f"scale={scaling_kind:>6s} "
                f"R=2 σ={sigma if sigma is not None else '-'} "
                f"λ1={lambda1:.4g} λ2={lambda2:.4g} "
                f"c={coupling:.4g} "
                f"α1={alpha1:.4g} α2={alpha2:.4g} "
                f"dy={dyadic_order} C={C:.4g} "
                f"| gram={gram_time:7.3f}s cv={fit_time:7.3f}s "
                f"| score={score:.4f}"
            )

        del G_train, svc, scores
        gc.collect()
        maybe_cleanup(trial.number)

        return score

    return objective


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    trained_models = {}
    final_results = {}

    datasets = tqdm(_datasets, position=0, leave=True)

    for name in datasets:
        x_train_raw, y_train_raw, x_test_raw, y_test_raw = UCR_UEA_datasets(use_cache=True).load_dataset(name)

        if (
            x_train_raw is None
            or y_train_raw is None
            or x_test_raw is None
            or y_test_raw is None
        ):
            print(f"Skipping {name}: one of the dataset splits is None.")
            continue

        for (at, ll) in TRANSFORMS:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test = le.transform(y_test_raw)

            x_train_torch_by_scaling = {}
            x_test_torch_by_scaling = {}
            train_shape_by_scaling = {}
            test_shape_by_scaling = {}
            subsample_by_scaling = {}

            for scaling_kind in SCALING_METHODS:
                x_train_scaled, x_test_scaled = scale_train_test(
                    x_train_raw,
                    x_test_raw,
                    scaling_kind=scaling_kind,
                )

                x_train = SIG.transform(x_train_scaled, at=at, ll=ll, scale=0.1)
                x_test = SIG.transform(x_test_scaled, at=at, ll=ll, scale=0.1)

                subsample = max(int(np.floor(x_train.shape[1] / 149)), 1)

                x_train = x_train[:, ::subsample, :]
                x_test = x_test[:, ::subsample, :]

                device_vol, dtype_vol = choose_device_and_dtype_volterra(x_train)

                x_train_torch_by_scaling[scaling_kind] = torch.tensor(
                    x_train,
                    dtype=dtype_vol,
                    device=device_vol,
                )
                x_test_torch_by_scaling[scaling_kind] = torch.tensor(
                    x_test,
                    dtype=dtype_vol,
                    device=device_vol,
                )

                train_shape_by_scaling[scaling_kind] = list(x_train.shape)
                test_shape_by_scaling[scaling_kind] = list(x_test.shape)
                subsample_by_scaling[scaling_kind] = subsample

                print(
                    f"[prepared] dataset={name} scaling={scaling_kind} "
                    f"train={x_train.shape} test={x_test.shape} subsample={subsample}"
                )

            warmup_scaling_kind = SCALING_METHODS[0]
            x_train_torch_warmup = x_train_torch_by_scaling[warmup_scaling_kind]

            datasets.set_description(
                f"dataset: {name} --- train {train_shape_by_scaling[warmup_scaling_kind]} "
                f"test {test_shape_by_scaling[warmup_scaling_kind]}"
            )

            methods_pbar = tqdm(METHOD_SPECS, position=1, leave=False)

            for spec in methods_pbar:
                static_kernel_kind = spec["static_kernel_kind"]
                mkey = method_key(static_kernel_kind)

                methods_pbar.set_description(f"{name} | {mkey}")

                if VSIG_OUTSIDE_WARMUP:
                    print(f"\n[warmup] dataset={name} method={mkey}")
                    t_warm0 = time.perf_counter()

                    warmup_vsig_all_ranks_for_dataset_shape(
                        x_train_torch=x_train_torch_warmup,
                        static_kernel_kind=static_kernel_kind,
                    )

                    t_warm1 = time.perf_counter()
                    warmup_time = t_warm1 - t_warm0
                    print(f"[timing] outside warmup (R=1 and R=2): {warmup_time:.3f} sec")

                else:
                    warmup_time = None

                sampler = TPESampler(
                    seed=OPTUNA_SEED,
                    n_startup_trials=OPTUNA_STARTUP_TRIALS,
                )

                study = optuna.create_study(direction="maximize", sampler=sampler)

                objective = make_volterra_objective(
                    x_train_torch_by_scaling=x_train_torch_by_scaling,
                    y_train=y_train,
                    static_kernel_kind=static_kernel_kind,
                )

                print(
                    f"\n[optuna] dataset={name}, kernel={mkey}, "
                    f"scaling optimized over={SCALING_METHODS}, "
                    f"trials={OPTUNA_N_TRIALS}"
                )

                t_optuna0 = time.perf_counter()

                study.optimize(
                    objective,
                    n_trials=OPTUNA_N_TRIALS,
                    gc_after_trial=True,
                    show_progress_bar=False,
                )

                t_optuna1 = time.perf_counter()
                optuna_time = t_optuna1 - t_optuna0

                completed_trials = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]

                if len(completed_trials) == 0:
                    print(
                        f"[warning] no successful trial for "
                        f"dataset={name}, kernel={mkey}"
                    )
                    maybe_cleanup(force=True)
                    continue

                best_params = dict(study.best_trial.params)
                train_cv_score = float(study.best_value)
                best_state_rank = int(best_params["state_rank"])
                best_scaling_kind = str(best_params["scaling_kind"])

                x_train_torch_best = x_train_torch_by_scaling[best_scaling_kind]
                x_test_torch_best = x_test_torch_by_scaling[best_scaling_kind]

                print("\n[optuna best]")
                print(f"  dataset     : {name}")
                print(f"  kernel      : {mkey}")
                print(f"  best scaling: {best_scaling_kind}")
                print(f"  best R      : {best_state_rank}")
                print(f"  best score  : {train_cv_score:.4f}")
                print(f"  best params : {best_params}")

                # --------------------------------------------------------
                # Refit on raw train Gram using best scaling
                # --------------------------------------------------------
                t_refit0 = time.perf_counter()

                with torch.no_grad():
                    G_train_best, resolved_params = compute_vsig_train_gram(
                        x_train_torch=x_train_torch_best,
                        state_rank=best_state_rank,
                        static_kernel_kind=static_kernel_kind,
                        dyadic_order=int(best_params["dyadic_order"]),
                        sigma=None if "sigma" not in best_params else float(best_params["sigma"]),

                        # R=1
                        lambda_base=None if "lambda_base" not in best_params else float(best_params["lambda_base"]),
                        alpha_scale=None if "alpha_scale" not in best_params else float(best_params["alpha_scale"]),

                        # R=2
                        lambda1=None if "lambda1" not in best_params else float(best_params["lambda1"]),
                        lambda2=None if "lambda2" not in best_params else float(best_params["lambda2"]),
                        coupling=None if "coupling" not in best_params else float(best_params["coupling"]),
                        alpha1=None if "alpha1" not in best_params else float(best_params["alpha1"]),
                        alpha2=None if "alpha2" not in best_params else float(best_params["alpha2"]),
                    )

                resolved_params["scaling_kind"] = best_scaling_kind

                raw_train_stats = kernel_health_stats(G_train_best)

                if not gram_is_usable(G_train_best, max_abs=GRAM_MAX_ABS):
                    print(f"[warning] best-trial train Gram unusable in refit: {raw_train_stats}")
                    maybe_cleanup(force=True)
                    continue

                final_estimator = SVC(
                    C=float(best_params["C"]),
                    kernel="precomputed",
                    decision_function_shape="ovo",
                )

                try:
                    final_estimator.fit(G_train_best, y_train)

                except Exception as e:
                    print(
                        f"[warning] final refit failed for "
                        f"dataset={name}, kernel={mkey}: {repr(e)}"
                    )
                    print(f"[warning] raw train kernel stats: {raw_train_stats}")
                    del G_train_best, final_estimator, study
                    gc.collect()
                    maybe_cleanup(force=True)
                    continue

                t_refit1 = time.perf_counter()
                refit_time = t_refit1 - t_refit0

                # --------------------------------------------------------
                # Test on raw test-vs-train Gram using best scaling
                # --------------------------------------------------------
                t_test0 = time.perf_counter()

                with torch.no_grad():
                    G_test, _ = compute_vsig_test_gram(
                        x_train_torch=x_train_torch_best,
                        x_test_torch=x_test_torch_best,
                        state_rank=best_state_rank,
                        static_kernel_kind=static_kernel_kind,
                        dyadic_order=int(best_params["dyadic_order"]),
                        sigma=None if "sigma" not in best_params else float(best_params["sigma"]),

                        # R=1
                        lambda_base=None if "lambda_base" not in best_params else float(best_params["lambda_base"]),
                        alpha_scale=None if "alpha_scale" not in best_params else float(best_params["alpha_scale"]),

                        # R=2
                        lambda1=None if "lambda1" not in best_params else float(best_params["lambda1"]),
                        lambda2=None if "lambda2" not in best_params else float(best_params["lambda2"]),
                        coupling=None if "coupling" not in best_params else float(best_params["coupling"]),
                        alpha1=None if "alpha1" not in best_params else float(best_params["alpha1"]),
                        alpha2=None if "alpha2" not in best_params else float(best_params["alpha2"]),
                    )

                raw_test_stats = kernel_health_stats(G_test)

                if not gram_is_usable(G_test, max_abs=GRAM_MAX_ABS):
                    print(
                        f"[warning] test Gram unusable for "
                        f"dataset={name}, kernel={mkey}: {raw_test_stats}"
                    )
                    del G_train_best, G_test, final_estimator, study
                    gc.collect()
                    maybe_cleanup(force=True)
                    continue

                try:
                    test_score = float(final_estimator.score(G_test, y_test))

                except Exception as e:
                    print(
                        f"[warning] test scoring failed for "
                        f"dataset={name}, kernel={mkey}: {repr(e)}"
                    )
                    print(f"[warning] raw test kernel stats: {raw_test_stats}")
                    del G_train_best, G_test, final_estimator, study
                    gc.collect()
                    maybe_cleanup(force=True)
                    continue

                t_test1 = time.perf_counter()
                test_time = t_test1 - t_test0

                print(
                    f"[result] dataset={name} kernel={mkey} "
                    f"best_scaling={best_scaling_kind} "
                    f"| best_R={best_state_rank} "
                    f"train_cv={train_cv_score:.4f} "
                    f"test={test_score:.4f}"
                )

                record = {
                    "dataset": name,
                    "method_key": mkey,
                    "scaling_kind": best_scaling_kind,
                    "state_rank": best_state_rank,
                    "static_kernel_kind": static_kernel_kind,
                    "transform": {
                        "time_augmentation": at,
                        "lead_lag": ll,
                    },
                    "subsample": subsample_by_scaling[best_scaling_kind],
                    "train_shape": train_shape_by_scaling[best_scaling_kind],
                    "test_shape": test_shape_by_scaling[best_scaling_kind],
                    "best_params": best_params,
                    "resolved_params": resolved_params,
                    "train_cv_score": train_cv_score,
                    "test_score": test_score,
                    "refit_C": float(best_params["C"]),
                    "timings": {
                        "warmup_time": warmup_time,
                        "optuna_time": optuna_time,
                        "refit_time": refit_time,
                        "test_time": test_time,
                    },
                    "estimator": final_estimator,
                    "label_encoder_classes": to_builtin(le.classes_),
                    "study_trials": study_to_records(study),
                    "run_tag": RUN_TAG,
                    "hostname": socket.gethostname(),
                }

                save_completed_result(record)

                trained_models[(name, mkey)] = record

                final_results[(name, mkey)] = {
                    "scaling_kind": best_scaling_kind,
                    "training_accuracy": train_cv_score,
                    "testing_accuracy": test_score,
                    "best_params": to_builtin(best_params),
                }

                del G_train_best, G_test, final_estimator, study
                gc.collect()
                maybe_cleanup(force=True)

            del x_train_torch_by_scaling, x_test_torch_by_scaling
            gc.collect()
            maybe_cleanup(force=True)

    print("\n[done]")
    for key, val in final_results.items():
        print(key, val)

    if MERGE_SUMMARIES_AT_END:
        merge_saved_summaries(OUTPUT_DIR)