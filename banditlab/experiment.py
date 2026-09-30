"""Compare bandit policies in the semi-synthetic simulator and log the runs to MLflow."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from banditlab.bandit import ACTIONS, CONTEXTS, EpsilonGreedy, ThompsonSampling
from banditlab.data import ROOT, load_data

LABELS = {
    "fixed": "Regra fixa: melhor global no piloto",
    "segmented": "Regra fixa por contexto",
    "thompson": "Thompson Sampling",
    "epsilon_greedy": "Epsilon-Greedy",
    "oracle": "Oráculo do simulador (limite superior)",
}


def build_environment(frame: pd.DataFrame, config: dict) -> dict:
    """Calibrate per-context reward probabilities from the earliest historical labels."""
    cut = int(len(frame) * config["training_fraction"])
    test_start = int(len(frame) * (config["training_fraction"] + config["calibration_fraction"]))
    train = frame.iloc[:cut]
    validation = frame.iloc[cut:test_start]
    test = frame.iloc[test_start:]
    grouped = train.groupby("contact")["converted"].agg(["sum", "count"])
    rates = np.array([(grouped.loc[c, "sum"] + 1) / (grouped.loc[c, "count"] + 2) for c in CONTEXTS])
    test_weights = test.contact.value_counts(normalize=True).reindex(CONTEXTS, fill_value=0).to_numpy()
    pilot_weights = validation.contact.value_counts(normalize=True).reindex(CONTEXTS, fill_value=0).to_numpy()
    return {
        "reference_rates": rates.tolist(),
        "probabilities": np.clip(
            rates[:, None] * np.array(config["assumed_message_multipliers"]), 0.001, 0.999
        ).tolist(),
        "observed_test_context_weights": test_weights.tolist(),
        "observed_pilot_context_weights": pilot_weights.tolist(),
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "calibration": "Beta(1,1) smoothing on training historical conversions; hypothetical action multipliers",
        "test_labels_used_for_rewards": False,
    }


def pilot_state(probabilities, weights, steps, seed):
    """Run the random pilot that every policy starts from."""
    rng = np.random.default_rng(seed)
    contexts = rng.choice(2, steps, p=weights)
    actions = rng.integers(3, size=steps)
    rewards = rng.random(steps) < probabilities[contexts, actions]
    alpha, beta = np.ones((2, 3)), np.ones((2, 3))
    np.add.at(alpha, (contexts, actions), rewards)
    np.add.at(beta, (contexts, actions), 1 - rewards)
    means = alpha / (alpha + beta)
    global_means = alpha.sum(axis=0) / (alpha + beta).sum(axis=0)
    return alpha, beta, int(global_means.argmax()), means.argmax(axis=1)


def simulate_once(probabilities, weights, config, seed, pilot_weights=None):
    """Run all policies for one seed on the same contexts and potential outcomes."""
    n = config["steps"]
    streams = np.random.SeedSequence(seed).spawn(4)
    rng = np.random.default_rng(streams[0])
    contexts = rng.choice(2, n, p=weights)
    potential_rewards = rng.random((n, 3)) < probabilities[contexts]
    alpha, beta, fixed, segmented = pilot_state(
        probabilities,
        weights if pilot_weights is None else pilot_weights,
        config["pilot_steps"],
        streams[1],
    )
    paths = {
        "fixed": np.full(n, fixed),
        "segmented": segmented[contexts],
        "oracle": probabilities[contexts].argmax(axis=1),
    }
    policies = {
        "thompson": ThompsonSampling(streams[2], alpha=alpha, beta=beta),
        "epsilon_greedy": EpsilonGreedy(streams[3], epsilon=config["epsilon"], alpha=alpha, beta=beta),
    }
    exploration = {}
    for name, policy in policies.items():
        actions = np.empty(n, dtype=int)
        non_greedy = 0
        for i, context in enumerate(contexts):
            scores = policy.means[context]
            action = policy.select(context)
            non_greedy += int(scores[action] < scores.max())
            reward = int(potential_rewards[i, action])
            policy.update(context, action, reward)
            actions[i] = action
        paths[name] = actions
        exploration[name] = non_greedy / n
    metrics, traces, allocation = [], [], []
    index = np.arange(n)
    optimal = probabilities[contexts].max(axis=1)
    for name, actions in paths.items():
        rewards = potential_rewards[index, actions]
        expected = probabilities[contexts, actions]
        regret = optimal - expected
        metrics.append(
            {
                "seed": seed,
                "policy": name,
                "conversion": float(rewards.mean()),
                "expected_conversion": float(expected.mean()),
                "conversions": int(rewards.sum()),
                "expected_regret": float(regret.sum()),
                "non_greedy_share": exploration.get(name, 0.0),
                "steps": n,
                "pilot_fixed_action": ACTIONS[fixed],
            }
        )
        checkpoints = np.unique(np.r_[np.arange(199, n, 200), n - 1])
        cumulative_rewards = rewards.cumsum()
        cumulative_regret = regret.cumsum()
        for i in checkpoints:
            traces.append(
                {
                    "seed": seed,
                    "policy": name,
                    "step": int(i + 1),
                    "conversion": float(cumulative_rewards[i] / (i + 1)),
                    "expected_regret": float(cumulative_regret[i]),
                }
            )
        for c, context in enumerate(CONTEXTS):
            selected = actions[contexts == c]
            for a, action in enumerate(ACTIONS):
                allocation.append(
                    {
                        "seed": seed,
                        "policy": name,
                        "context": context,
                        "action": action,
                        "count": int((selected == a).sum()),
                        "context_count": len(selected),
                    }
                )
    return metrics, traces, allocation, policies["thompson"].state()


def bootstrap_mean_ci(values, seed=2026):
    """Bootstrap 95% confidence interval of the mean across seeds."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), *[float(v) for v in np.quantile(means, [0.025, 0.975])]


def summarize(metrics: pd.DataFrame):
    rows = []
    for (scenario, policy), group in metrics.groupby(["scenario", "policy"]):
        mean, lo, hi = bootstrap_mean_ci(group.conversion)
        rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "conversion": mean,
                "ci_low": lo,
                "ci_high": hi,
                "expected_regret": float(group.expected_regret.mean()),
                "non_greedy_share": float(group.non_greedy_share.mean()),
            }
        )
    comparisons = []
    for scenario, group in metrics.groupby("scenario"):
        pivot = group.pivot(index="seed", columns="policy", values="conversion")
        for baseline in ("fixed", "segmented"):
            difference = pivot.thompson - pivot[baseline]
            gain, lo, hi = bootstrap_mean_ci(difference)
            comparisons.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "gain_pp": 100 * gain,
                    "ci_low_pp": 100 * lo,
                    "ci_high_pp": 100 * hi,
                    "relative_lift_percent": 100 * gain / float(pivot[baseline].mean()),
                    "win_fraction": float((difference > 0).mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def plot_results(summary, traces, allocations, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    palette = {
        "fixed": "#8093A4",
        "segmented": "#7662A8",
        "thompson": "#087E8B",
        "epsilon_greedy": "#DB8E30",
        "oracle": "#303C4A",
    }
    names = ["fixed", "segmented", "epsilon_greedy", "thompson", "oracle"]
    selected = summary[summary.scenario == "contextual"].set_index("policy").loc[names]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    y = selected.conversion.to_numpy() * 100
    errors = np.array([y - selected.ci_low.to_numpy() * 100, selected.ci_high.to_numpy() * 100 - y])
    bars = ax.barh(range(5), y, xerr=errors, color=[palette[k] for k in names], capsize=4)
    ax.set_yticks(range(5), [LABELS[k] for k in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(y) * 1.2)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in y], padding=12)
    ax.set_xlabel("Conversão simulada (%) | IC 95% da média entre sementes")
    ax.set_title("Mensagens adaptativas | cenário hipotético com diferenças por canal", loc="left", pad=18)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    mean_trace = (
        traces[traces.scenario == "contextual"]
        .groupby(["policy", "step"])[["conversion", "expected_regret"]]
        .mean()
    )
    for name in names:
        data = mean_trace.loc[name]
        axes[0].plot(data.index, 100 * data.conversion, color=palette[name], label=LABELS[name])
        axes[1].plot(data.index, data.expected_regret, color=palette[name])
    axes[0].set(ylabel="Conversão acumulada simulada (%)", xlabel="Decisões após piloto")
    axes[1].set(ylabel="Arrependimento esperado acumulado", xlabel="Decisões após piloto")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Aprendizado online | média das repetições", x=0.05, ha="left")
    fig.savefig(output / "learning.png", dpi=170)
    plt.close(fig)

    chosen = allocations[(allocations.scenario == "contextual") & (allocations.policy == "thompson")]
    table = (
        chosen.groupby(["context", "action"])["count"]
        .sum()
        .unstack()
        .reindex(index=CONTEXTS, columns=ACTIONS)
    )
    shares = table.div(table.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    shares.plot.barh(stacked=True, ax=ax, color=["#087E8B", "#DB8E30", "#8093A4"])
    ax.set(
        xlabel="Proporção de decisões",
        ylabel="Canal histórico",
        title="Thompson Sampling: distribuição das mensagens (simulação)",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3, fontsize=8)
    fig.savefig(output / "allocation.png", dpi=170)
    plt.close(fig)


def run_experiments(config_path: Path = ROOT / "config.json", tracking=True):
    config = json.loads(config_path.read_text())
    if config["repetitions"] < 2 or config["steps"] < 1 or config["pilot_steps"] < 1:
        raise ValueError("Use at least two repetitions and positive simulation lengths")
    if config["prior_alpha"] != 1 or config["prior_beta"] != 1:
        raise ValueError("This implementation supports only the documented Beta(1,1) prior")
    frame = load_data()
    environment = build_environment(frame, config)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    metrics, traces, allocations = [], [], []
    snapshot = None
    for scenario in config["scenarios"]:
        probabilities = np.array(environment["probabilities"])
        weights = config["context_weights"]
        pilot_weights = None
        if scenario == "equal_arms":
            probabilities = np.repeat(np.array(environment["reference_rates"])[:, None], 3, axis=1)
        elif scenario == "observed_mix":
            weights = environment["observed_test_context_weights"]
            pilot_weights = environment["observed_pilot_context_weights"]
        elif scenario != "contextual":
            raise ValueError(f"Unknown scenario: {scenario}")
        for seed in range(config["first_seed"], config["first_seed"] + config["repetitions"]):
            m, t, a, state = simulate_once(probabilities, weights, config, seed, pilot_weights)
            for values, destination in ((m, metrics), (t, traces), (a, allocations)):
                destination.extend({"scenario": scenario, **v} for v in values)
            if scenario == "contextual" and seed == config["first_seed"]:
                snapshot = {
                    "algorithm": "segmented_thompson_sampling",
                    "contexts": list(CONTEXTS),
                    "actions": list(ACTIONS),
                    "training_kind": "semi_synthetic",
                    "seed": seed,
                    "simulation_version": config["simulation_version"],
                    **state,
                }
        print(f"Completed {scenario}: {config['repetitions']} seeds", flush=True)
    metrics, traces, allocations = map(pd.DataFrame, (metrics, traces, allocations))
    summary, comparisons = summarize(metrics)
    for name, table in [
        ("metrics", metrics),
        ("traces", traces),
        ("allocations", allocations),
        ("summary", summary),
        ("comparisons", comparisons),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    plot_results(summary, traces, allocations, output)
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    if snapshot is not None:
        snapshot["version"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()[:16]
        (models / "policy.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    if tracking:
        log_experiments(config, metrics, comparisons, output)
    return summary, comparisons


def log_experiments(config, metrics, comparisons, output):
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    name = config["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(name)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    source_paths = sorted((ROOT / "banditlab").glob("*.py")) + [ROOT / "config.json"]
    code_digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in source_paths)).hexdigest()
    run_rows = []
    with mlflow.start_run(run_name="complete-evaluation") as parent:
        mlflow.log_params(
            {
                "steps": config["steps"],
                "repetitions": config["repetitions"],
                "pilot_steps": config["pilot_steps"],
                "epsilon": config["epsilon"],
                "prior": "Beta(1,1)",
                "first_seed": config["first_seed"],
                "dataset_sha256": manifest["source_csv_sha256"],
                "code_sha256": code_digest,
                "simulation_version": config["simulation_version"],
            }
        )
        mlflow.set_tags({"evaluation": "semi_synthetic", "real_world_uplift_established": "false"})
        for row in comparisons.to_dict("records"):
            prefix = f"{row['scenario']}_{row['baseline']}"
            mlflow.log_metrics(
                {f"{prefix}_{key}": row[key] for key in ("gain_pp", "ci_low_pp", "ci_high_pp")}
            )
        for row in metrics.to_dict("records"):
            with mlflow.start_run(
                run_name=f"{row['scenario']}-{row['policy']}-{row['seed']}", nested=True
            ) as run:
                mlflow.log_params(
                    {k: row[k] for k in ("scenario", "policy", "seed", "steps", "pilot_fixed_action")}
                )
                mlflow.log_metrics(
                    {
                        k: row[k]
                        for k in ("conversion", "expected_conversion", "expected_regret", "non_greedy_share")
                    }
                )
                run_rows.append(
                    {
                        "run_id": run.info.run_id,
                        "scenario": row["scenario"],
                        "policy": row["policy"],
                        "seed": row["seed"],
                    }
                )
        for path in (ROOT / "config.json", ROOT / "data/manifest.json", ROOT / "models/policy.json"):
            mlflow.log_artifact(str(path))
        for path in source_paths:
            mlflow.log_artifact(str(path), "source")
        for path in output.iterdir():
            if path.is_file() and path.name != "tracking.json":
                mlflow.log_artifact(str(path), "evaluation")
        for path in (ROOT / "requirements.txt", ROOT / "requirements-lock.txt"):
            mlflow.log_artifact(str(path), "source")
        tracking_path = output / "tracking.json"
        tracking_path.write_text(
            json.dumps(
                {"experiment": name, "parent_run_id": parent.info.run_id, "child_runs": run_rows}, indent=2
            )
            + "\n"
        )
        mlflow.log_artifact(str(tracking_path), "evaluation")


if __name__ == "__main__":
    summary, comparisons = run_experiments()
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
