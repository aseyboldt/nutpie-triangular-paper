import pymc as pm
import numpy as np
import nutpie
import pandas as pd
import seaborn as sns
import jax.numpy as jnp
import jax
import asdex
import sksparse
import matplotlib.pyplot as plt
import arviz
import time
import sys

sys.path.append("/root/pymc-model-catalogue/")


def models():
    def make_stan(name, *, ad_hessian=True):
        data, model = name.split("-", maxsplit=1)

        import posteriordb

        db = posteriordb.PosteriorDatabase("../posteriordb/posterior_database")

        compiled = nutpie.compile_stan_model(
            filename=db.model(model).code_file_path("stan"),
            cache=True,
            ad_hessian=ad_hessian,
        )
        compiled = compiled.with_data(**db.data(data).values())

        return name, compiled

    names = [
        "diamonds-diamonds",
        "Mth_data-Mth_model",
        "eight_schools-eight_schools_centered",
        "eight_schools-eight_schools_noncentered",
        "arK-arK",
        "garch-garch11",
        "butterfly-multi_occupancy",
        "GLMM_Poisson_data-GLMM_Poisson_model",
        "gp_pois_regr-gp_pois_regr",
        "mcycle_gp-accel_gp",
        "mcycle_splines-accel_splines",
        "normal_2-normal_mixture",
        "normal_5-normal_mixture_k",
        "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse",
        "radon_mn-radon_hierarchical_intercept_centered",
        "radon_mn-radon_variable_intercept_slope_centered",
        "radon_mn-radon_variable_slope_centered",
        "surgical_data-surgical_model",
    ]

    for name in names:
        name, model = make_stan(name)
        model = model.with_hessian_sparsity()
        model = model.with_factorization()
        yield name, model

    name, model = make_stan("pilots-pilots")
    model = model.with_hessian_sparsity()
    model = model.with_factorization(front=["sigma_a", "sigma_b", "sigma_y", "mu_a", "mu_b"])
    yield name, model

    #name = "arma-arma11"  # initialization failure at seed=50

    names = [
        "one_comp_mm_elim_abs-one_comp_mm_elim_abs",
        "soil_carbon-soil_incubation",
    ]

    for name in names:
        name, model = make_stan(name, ad_hessian=False)
        model = model.with_hessian_sparsity(np.ones((model.n_dim, model.n_dim)))
        model = model.with_factorization()
        yield name, model


    name, model = make_stan("election88-election88_full")
    model = (
        model
        .with_hessian_sparsity()
        .with_factorization(front=["sigma_a", "sigma_b", "sigma_c", "sigma_d"])
    )
    yield name, model

    name, model = make_stan("sat-hier_2pl")
    model = model.with_hessian_sparsity(num_points=5, seed=5)
    model = model.with_factorization()
    yield name, model

    name, model = make_stan("uk_drivers-state_space_stochastic_level_stochastic_seasonal")
    model = model.with_hessian_sparsity(num_points=5, seed=5)
    model = model.with_factorization(front=["sigma", "beta", "lambda"])
    yield name, model

    import models.stochastic_volatility
    model, _ = models.stochastic_volatility.build_model()

    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    compiled = compiled.with_hessian_sparsity(num_points=5, seed=5)

    yield "stochastic_volatility", compiled

def evaluate(name, model, **kwargs):
    start = time.time()
    tr = nutpie.sample(model, **kwargs)
    end = time.time()

    ess = arviz.ess(tr).min().to_dataset().to_pandas().min()
    tail_ess = arviz.ess(tr, method="tail").min().to_dataset().to_pandas().min()
    warmup_grads = tr.warmup_sample_stats.n_gradients.sum()
    sampling_grads = tr.sample_stats.n_gradients.sum()
    rhat = arviz.rhat(tr).max().to_dataset().to_pandas().max()

    return {
        "time": end - start,
        "ess": float(ess),
        "tail_ess": float(tail_ess),
        "rhat": float(rhat),
        "warmup_grads": int(warmup_grads),
        "sampling_grads": int(sampling_grads),
        "name": name,
        "divergences": int(tr.sample_stats.diverging.sum()),
        "n_dim": int(model.n_dim),
        "num_edges": model.factorization.num_edges,
        **kwargs,
    }


records = []
for name, model in models():
    for walnuts in [True, False]:
        for adaptation in ["diag", "flow"]:
            kwargs = {
                "adaptation": adaptation,
                "chains": 4,
                "tune": 3000,
                "progress_bar": False,
                "exact_normal_trajectory": True,
                "walnuts": walnuts,
                "maxdepth": 6 if walnuts else 10,
                "seed": 42,
            }
            if walnuts:
                kwargs.update({
                    "walnuts_max_step_halvings": 8,
                })
            try:
                record = evaluate(
                    name,
                    model.with_transform_adapt(scale_squash=8.0),
                    **kwargs,
                )
                records.append(record)
            except Exception as e:
                print(f"Error evaluating {name} with adaptation={adaptation}, walnuts={walnuts}: {e}")
                continue
            print(name, adaptation, record["ess"], record["divergences"], record["rhat"], record["time"])

pd.DataFrame.from_records(records).to_csv("bench.csv")
