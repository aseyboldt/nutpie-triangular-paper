"""Benchmark nutpie adaptation schemes on posteriordb and PyMC models.

Each (model, sampler settings, seed) run is cached with joblib.Memory. The
cache key includes a description of the machine and the relevant package
versions, so timings are never reused across machines or library versions.

Examples:

    python bench_models.py --list
    python bench_models.py --preset default -o results/bench.csv
    python bench_models.py --preset posteriordb-all --adaptation flow
    python bench_models.py --models stan:pilots-pilots stan:arK-arK --seeds 1 2 3
"""

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import os
import platform
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import arviz
import joblib
import numpy as np
import pandas as pd
import psutil
import watermark

import nutpie

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelConfig:
    """How to build one benchmark model.

    `sparsity` and `factorization` are passed on to `with_hessian_sparsity`
    and `with_factorization`. `factorization=None` skips the factorization.
    """

    name: str
    kind: str = "stan"
    ad_hessian: bool = True
    dense_sparsity: bool = False
    sparsity: dict = field(default_factory=dict)
    factorization: dict | None = field(default_factory=dict)


# posteriordb models that need settings other than the defaults. When adding
# more posteriordb models, put their settings here.
STAN_SETTINGS = {
    "pilots-pilots": {
        "factorization": {"front": ["sigma_a", "sigma_b", "sigma_y", "mu_a", "mu_b"]},
    },
    "one_comp_mm_elim_abs-one_comp_mm_elim_abs": {
        "ad_hessian": False,
        "dense_sparsity": True,
    },
    "soil_carbon-soil_incubation": {"ad_hessian": False, "dense_sparsity": True},
    "election88-election88_full": {
        "factorization": {"front": ["sigma_a", "sigma_b", "sigma_c", "sigma_d"]},
    },
    "sat-hier_2pl": {"sparsity": {"num_points": 5, "seed": 5}},
    "uk_drivers-state_space_stochastic_level_stochastic_seasonal": {
        "sparsity": {"num_points": 5, "seed": 5},
        "factorization": {"front": ["sigma", "beta", "lambda"]},
    },
}

# Settings for models from pymc-model-catalogue.
PYMC_SETTINGS = {
    "stochastic_volatility": {"sparsity": {"num_points": 5, "seed": 5}},
}


def stan(name):
    """A posteriordb model, with settings from STAN_SETTINGS."""
    return ModelConfig(name, kind="stan", **STAN_SETTINGS.get(name, {}))


# Settings for the PyMC-Marketing models from mmm-param-recovery. The model
# names are its dataset presets, see generate_mmm_data.py. asdex can't trace
# the adstock convolution, so unless the settings give a sparsity, these use
# the pattern from `mmm_hessian_sparsity`.
MMM_SETTINGS = {}


def pymc(name):
    """A pymc-model-catalogue model, with settings from PYMC_SETTINGS."""
    return ModelConfig(name, kind="pymc", **PYMC_SETTINGS.get(name, {}))


def mmm(name):
    """An mmm-param-recovery model, with settings from MMM_SETTINGS."""
    return ModelConfig(name, kind="mmm", **MMM_SETTINGS.get(name, {}))


MODEL_KINDS = {"stan": stan, "pymc": pymc, "mmm": mmm}

MMM_MODELS = [
    mmm("small_business"),
    mmm("medium_business"),
    mmm("large_business"),
    mmm("growing_business"),
]


# Models that did not converge in a previous paper and were excluded from its
# analysis. We test them explicitly to see if the new sampler handles them.
PREVIOUSLY_NONCONVERGED = [
    stan("diamonds-diamonds"),
    stan("Mth_data-Mth_model"),
    stan("eight_schools-eight_schools_centered"),
    stan("arK-arK"),
    stan("garch-garch11"),
    stan("butterfly-multi_occupancy"),
    stan("GLMM_Poisson_data-GLMM_Poisson_model"),
    stan("gp_pois_regr-gp_pois_regr"),
    stan("mcycle_gp-accel_gp"),
    stan("mcycle_splines-accel_splines"),
    stan("normal_2-normal_mixture"),
    stan("normal_5-normal_mixture_k"),
    stan("low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse"),
    stan("radon_mn-radon_hierarchical_intercept_centered"),
    stan("radon_mn-radon_variable_intercept_slope_centered"),
    stan("radon_mn-radon_variable_slope_centered"),
    stan("surgical_data-surgical_model"),
    stan("pilots-pilots"),
    stan("one_comp_mm_elim_abs-one_comp_mm_elim_abs"),
    stan("soil_carbon-soil_incubation"),
    stan("sat-hier_2pl"),
    stan("uk_drivers-state_space_stochastic_level_stochastic_seasonal"),
    # arma-arma11: initialization failure at seed=50
]

# Very slow or dense models that are never run as part of a preset.
SKIPPED_MODELS = [
    # TODO
]

PRESETS = {
    "default": lambda db: (
        PREVIOUSLY_NONCONVERGED
        + [
            stan("eight_schools-eight_schools_noncentered"),
            stan("election88-election88_full"),
            pymc("stochastic_volatility"),
        ]
    ),
    "previously-nonconverged": lambda db: PREVIOUSLY_NONCONVERGED,
    "posteriordb-all": lambda db: [stan(name) for name in sorted(db.posterior_names())],
    "mmm": lambda db: MMM_MODELS,
}


@dataclass(frozen=True)
class SamplerConfig:
    adaptation: str
    walnuts: bool
    chains: int
    tune: int
    draws: int

    def sample_kwargs(self, seed):
        kwargs = {
            "adaptation": self.adaptation,
            "walnuts": self.walnuts,
            "chains": self.chains,
            "tune": self.tune,
            "draws": self.draws,
            "seed": seed,
            "progress_bar": False,
            "exact_normal_trajectory": True,
            "maxdepth": 6 if self.walnuts else 10,
        }
        if self.walnuts:
            kwargs["walnuts_max_step_halvings"] = 8
        return kwargs


def machine_info():
    """Describe the machine and software, without anything time dependent.

    This is part of the cache key, so it must be stable between runs on the
    same machine.
    """
    packages = "nutpie,numpy,scipy,jax,pymc,pytensor,pymc_marketing,bridgestan,arviz"
    info = watermark.watermark(
        python=True, machine=True, hostname=True, packages=packages
    )
    # The size of rayon's thread pool, which the samplers run on, changes timings.
    rayon_threads = os.environ.get("RAYON_NUM_THREADS", "default")
    return f"{info}\nCPU model   : {cpu_model()}\nRAYON_NUM_THREADS: {rayon_threads}\n"


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def build_mmm_model(name, data_dir):
    """Build the PyMC-Marketing model of mmm-param-recovery for a dataset preset.

    The data comes from generate_mmm_data.py. This mirrors `build_pymc_model`
    and `prepare_dataset_for_modeling` in mmm_param_recovery/benchmarking,
    which we can't import directly because they pull in Meridian and
    TensorFlow, and are written for pymc-marketing 0.16. The contribution
    deterministics are left out, as they don't change the posterior.
    """
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturationSigmoid

    data = pd.read_csv(Path(data_dir) / f"{name}.csv", parse_dates=["date"])
    data = data.rename(columns={"date": "time"})
    data["population"] = 1
    channels = [col for col in data.columns if re.match(r"x\d+", col)]
    controls = [col for col in data.columns if re.match(r"^c\d+$", col)]

    spend = data.groupby("geo")[channels].sum().to_numpy()
    prior_sigma = len(channels) * spend / spend.sum(axis=1)[:, None]
    saturation = HillSaturationSigmoid(
        priors={
            "sigma": Prior(
                "InverseGamma",
                mu=Prior(
                    "HalfNormal", sigma=prior_sigma.mean(axis=0), dims=("channel",)
                ),
                sigma=Prior("HalfNormal", sigma=1.5),
                dims=("channel", "geo"),
            ),
            "beta": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
            "lam": Prior("HalfNormal", sigma=1.5, dims=("channel",)),
        },
    )
    model = MMM(
        date_column="time",
        target_column="y",
        channel_columns=channels,
        control_columns=controls,
        dims=("geo",),
        scaling={
            "channel": {"method": "max", "dims": ()},
            "target": {"method": "max", "dims": ()},
        },
        saturation=saturation,
        adstock=GeometricAdstock(
            l_max=8,
            priors={"alpha": Prior("Beta", alpha=1, beta=3, dims=("channel",))},
        ),
        yearly_seasonality=2,
    )
    model.build_model(X=data.drop(columns=["y"]), y=data["y"])
    return model.model


def mmm_hessian_sparsity(pymc_model, unconstrained_names):
    """The Hessian sparsity of an mmm-param-recovery model, from its structure.

    Each observation depends on the parameters of its geo and on the global
    (per channel) parameters, so two parameters interact unless they belong
    to different geos. `saturation_sigma_mu` and `saturation_sigma_sigma`
    only appear in the hierarchical prior of `saturation_sigma`, and each
    other prior is elementwise. We checked this against the full Hessian of
    small_business.
    """
    dims = {
        value.name: pymc_model.named_vars_to_dims.get(rv.name, ())
        for rv, value in pymc_model.rvs_to_values.items()
        if value in pymc_model.value_vars
    }

    # `unconstrained_names` are `{value_var}_{i.j}`, or `{value_var}` for scalars.
    params = []
    for label in unconstrained_names:
        if label in dims:
            params.append((label, {}))
            continue
        var, index = label.rsplit("_", 1)
        index = [int(i) for i in index.split(".")]
        assert len(index) == len(dims[var]), label
        params.append((var, dict(zip(dims[var], index))))

    mu, scale, sigma = (
        "saturation_sigma_mu_log__",
        "saturation_sigma_sigma_log__",
        "saturation_sigma_log__",
    )
    assert {mu, scale, sigma} <= dims.keys(), dims.keys()

    n = len(params)
    pattern = np.eye(n, dtype=bool)
    for i, (var_i, index_i) in enumerate(params):
        for j, (var_j, index_j) in enumerate(params[:i]):
            pair = {var_i, var_j}
            if pair & {mu, scale}:
                if pair == {mu, scale}:
                    interacts = True
                elif scale in pair:
                    interacts = sigma in pair
                else:
                    interacts = (
                        sigma in pair and index_i["channel"] == index_j["channel"]
                    )
            else:
                geo_i, geo_j = index_i.get("geo"), index_j.get("geo")
                interacts = geo_i is None or geo_j is None or geo_i == geo_j
            pattern[i, j] = pattern[j, i] = interacts
    return pattern


def build_model(model_config, paths):
    pattern = None
    if model_config.kind == "stan":
        import posteriordb

        db = posteriordb.PosteriorDatabase(str(paths["posteriordb"]))
        posterior = db.posterior(model_config.name)
        model = nutpie.compile_stan_model(
            filename=posterior.model.code_file_path("stan"),
            cache=True,
            ad_hessian=model_config.ad_hessian,
        )
        model = model.with_data(**posterior.data.values())
    elif model_config.kind == "pymc":
        import importlib

        sys.path.insert(0, str(paths["pymc"]))
        module = importlib.import_module(f"models.{model_config.name}")
        pymc_model, _ = module.build_model()
        model = nutpie.compile_pymc_model(
            pymc_model, backend="jax", gradient_backend="jax"
        )
    elif model_config.kind == "mmm":
        pymc_model = build_mmm_model(model_config.name, paths["mmm"])
        model = nutpie.compile_pymc_model(
            pymc_model, backend="jax", gradient_backend="jax"
        )
        if not model_config.sparsity:
            pattern = mmm_hessian_sparsity(pymc_model, model._unconstrained_names)
    else:
        raise ValueError(f"Unknown model kind {model_config.kind!r}")

    sparsity = model_config.sparsity
    if model_config.dense_sparsity:
        pattern = np.ones((model.n_dim, model.n_dim), dtype=bool)
    if pattern is not None:
        sparsity = {"sparsity": pattern, **sparsity}
    model = model.with_hessian_sparsity(**sparsity)
    if model_config.factorization is not None:
        model = model.with_factorization(**model_config.factorization)
    return model


class RunStopped(Exception):
    """A run was stopped because of its time or memory limit."""


def memory_usage():
    """The resident memory of this process and its children in bytes."""
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def sample_with_limits(model, kwargs, *, label, timeout, max_memory, interval=1.0):
    """Sample in the background, and cancel if a limit is exceeded.

    Raises `RunStopped` if sampling takes longer than `timeout` seconds, or
    the memory goes above `max_memory` (a fraction of the total RAM).
    Returns the trace and the peak memory in bytes.
    """
    max_bytes = max_memory * psutil.virtual_memory().total
    start = time.monotonic()
    peak = memory_usage()
    background = nutpie.sample(model, **kwargs, blocking=False)
    try:
        while True:
            # `wait` raises TimeoutError while sampling is still running.
            # (`is_finished` currently raises that error too.)
            try:
                trace = background.wait(timeout=interval)
                break
            except TimeoutError:
                pass
            memory = memory_usage()
            peak = max(peak, memory)
            elapsed = time.monotonic() - start
            reason = None
            if timeout is not None and elapsed > timeout:
                reason = f"timeout after {elapsed / 60:.1f} min"
            elif memory > max_bytes:
                reason = (
                    f"memory {memory / 2**30:.1f} GiB is above the limit of "
                    f"{max_bytes / 2**30:.1f} GiB"
                )
            if reason is not None:
                print(f"[stopped]  {label}: {reason}", flush=True)
                background.cancel()
                raise RunStopped(reason)
    except KeyboardInterrupt:
        background.cancel()
        raise
    return trace, max(peak, memory_usage())


def run(model_config, sampler, seed, machine, *, paths, timeout, max_memory):
    """Sample one model and summarize the trace.

    `machine` is not used, but is part of the joblib cache key. `paths` maps
    each model kind to the checkout it comes from. Sampling is stopped with
    `RunStopped` after `timeout` seconds or if it uses more than `max_memory`
    of the RAM.
    """
    model = build_model(model_config, paths)
    model = model.with_transform_adapt(scale_squash=8.0)
    kwargs = sampler.sample_kwargs(seed)

    start = time.perf_counter()
    tr, peak_memory = sample_with_limits(
        model,
        kwargs,
        label=f"{model_config.kind}:{model_config.name} {sampler.adaptation} seed={seed}",
        timeout=timeout,
        max_memory=max_memory,
    )
    end = time.perf_counter()

    ess = arviz.ess(tr).min().to_dataset().to_pandas().min()
    tail_ess = arviz.ess(tr, method="tail").min().to_dataset().to_pandas().min()
    rhat = arviz.rhat(tr).max().to_dataset().to_pandas().max()
    factorization = model.factorization

    return {
        "time": end - start,
        "ess": float(ess),
        "tail_ess": float(tail_ess),
        "ess_logp": float(arviz.ess(tr.sample_stats.logp)),
        "rhat": float(rhat),
        "warmup_grads": int(tr.warmup_sample_stats.n_gradients.sum()),
        "sampling_grads": int(tr.sample_stats.n_gradients.sum()),
        "divergences": int(tr.sample_stats.diverging.sum()),
        "n_dim": int(model.n_dim),
        "num_edges": None if factorization is None else factorization.num_edges,
        "peak_memory_gib": peak_memory / 2**30,
        **kwargs,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--preset", choices=PRESETS, default="default")
    selection.add_argument(
        "--models",
        nargs="+",
        metavar="KIND:NAME",
        help="Run only these models, e.g. stan:arK-arK or pymc:stochastic_volatility.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the selected models and exit."
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("bench.csv"))
    parser.add_argument(
        "--posteriordb",
        type=Path,
        default=HERE / "posteriordb" / "posterior_database",
        help="Path to the posterior_database directory of a posteriordb checkout.",
    )
    parser.add_argument(
        "--model-catalogue",
        type=Path,
        default=HERE / "pymc-model-catalogue",
        help="Path to a pymc-model-catalogue checkout (for pymc: models).",
    )
    parser.add_argument(
        "--mmm-data",
        type=Path,
        default=HERE / "data" / "mmm",
        help="Directory with the datasets from generate_mmm_data.py (for mmm: models).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=HERE / ".bench-cache",
        help="joblib.Memory directory for caching model runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rerun everything and don't write to the cache.",
    )
    parser.add_argument(
        "--cache-tag",
        default="",
        help="Extra string for the cache key, e.g. to force reruns after rebuilding nutpie.",
    )
    parser.add_argument("--adaptation", nargs="+", default=["diag", "flow"])
    parser.add_argument(
        "--walnuts", nargs="+", choices=["on", "off"], default=["on", "off"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--tune", type=int, default=3000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=20,
        help="Stop sampling a model after this many minutes.",
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        default=0.7,
        help="Stop a run if it uses more than this fraction of the total RAM.",
    )
    return parser.parse_args(argv)


def select_models(args):
    import posteriordb

    db = posteriordb.PosteriorDatabase(str(args.posteriordb))
    if args.models is None:
        return [
            model for model in PRESETS[args.preset](db) if model not in SKIPPED_MODELS
        ]
    return [parse_model(model) for model in args.models]


def parse_model(arg):
    kind, sep, name = arg.partition(":")
    if not sep or kind not in MODEL_KINDS:
        raise SystemExit(
            f"Invalid model {arg!r}, expected one of {list(MODEL_KINDS)} as KIND:NAME"
        )
    return MODEL_KINDS[kind](name)


class WarningCounter:
    """Collect warnings instead of printing them, to report them at the end."""

    def __init__(self):
        self.counts = collections.Counter()

    def add(self, message, category, filename, lineno, file=None, line=None):
        self.counts[(category.__name__, str(message), filename, lineno)] += 1

    def report(self):
        if not self.counts:
            return
        print(f"\n{len(self.counts)} distinct warnings:")
        for (category, message, filename, lineno), count in self.counts.most_common():
            first_line = message.splitlines()[0] if message else ""
            print(f"  {count:>6}x {category}: {first_line}")
            print(f"          at {filename}:{lineno}")


def main(argv=None):
    args = parse_args(argv)
    counter = WarningCounter()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = counter.add
        try:
            interrupted = benchmark(args)
        finally:
            counter.report()
    if interrupted:
        sys.exit(130)


def benchmark(args):
    """Run the benchmark. Returns whether it was interrupted."""
    model_configs = select_models(args)

    if args.list:
        for model_config in model_configs:
            print(f"{model_config.kind}:{model_config.name}")
        return False

    machine = machine_info() + args.cache_tag
    print(machine)

    memory = joblib.Memory(None if args.no_cache else args.cache_dir, verbose=0)
    # Results are valid regardless of the limits, as long as the run finished.
    cached_run = memory.cache(run, ignore=["paths", "timeout", "max_memory"])
    paths = {
        "posteriordb": args.posteriordb,
        "pymc": args.model_catalogue,
        "mmm": args.mmm_data,
    }

    samplers = [
        SamplerConfig(adaptation, walnuts == "on", args.chains, args.tune, args.draws)
        for walnuts in args.walnuts
        for adaptation in args.adaptation
    ]

    records = []
    interrupted = False
    runs = itertools.product(model_configs, samplers, args.seeds)
    for model_config, sampler, seed in runs:
        meta = {
            "kind": model_config.kind,
            "name": model_config.name,
            "seed": seed,
            **dataclasses.asdict(sampler),
        }
        run_args = (model_config, sampler, seed, machine)
        run_kwargs = {
            "paths": paths,
            "timeout": args.timeout * 60,
            "max_memory": args.max_memory,
        }
        cached = cached_run.check_call_in_cache(*run_args, **run_kwargs)
        try:
            record = cached_run(*run_args, **run_kwargs)
        except KeyboardInterrupt:
            print(f"Interrupted in {meta}, writing the results so far")
            records.append({**meta, "error": "interrupted"})
            interrupted = True
            break
        except RunStopped as e:
            records.append({**meta, "error": f"stopped: {e}"})
            continue
        except Exception as e:
            traceback.print_exc()
            print(f"Error in {meta}: {e}")
            records.append({**meta, "error": repr(e)})
            continue
        records.append({**meta, **record, "cached": cached, "error": None})
        print(
            "[cached]  " if cached else "[computed]",
            model_config.name,
            sampler.adaptation,
            f"walnuts={sampler.walnuts}",
            f"seed={seed}",
            f"ess={record['ess']:.1f}",
            f"div={record['divergences']}",
            f"rhat={record['rhat']:.3f}",
            f"time={record['time']:.1f}s",
        )

    machine_hash = hashlib.sha256(machine.encode()).hexdigest()[:12]
    df = pd.DataFrame.from_records(records).assign(machine=machine_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "machine": machine,
                "machine_hash": machine_hash,
                "argv": sys.argv,
                "interrupted": interrupted,
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output} and {meta_path}")
    return interrupted


if __name__ == "__main__":
    main()
