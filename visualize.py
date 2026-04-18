"""
visualize.py — Skill Map Generator
===================================
Generates a 2D scatter plot of all skills in PCA latent space,
overlaid with the candidate's position and highlighted recommendations.

Usage
-----
  # From Python
  from visualize import plot_skill_map
  plot_skill_map(
      analyzer,
      candidate_skills=["tableau", "power_bi", "excel"],
      save_path="my_skill_map.png"   # omit to just show interactively
  )

  # From CLI
  python visualize.py --skills "tableau,power_bi,excel" --out map.png
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

# ── Skill → domain category mapping ──────────────────────────────────────────
DOMAIN_COLORS = {
    "Programming":      "#4E9AF1",   # blue
    "ML / DL":          "#E84393",   # pink
    "NLP":              "#9B59B6",   # purple
    "Computer Vision":  "#8E44AD",   # dark purple
    "Data Processing":  "#27AE60",   # green
    "Visualisation":    "#F39C12",   # orange
    "Cloud / MLOps":    "#E74C3C",   # red
    "Databases":        "#16A085",   # teal
    "Math / Stats":     "#2980B9",   # dark blue
    "Web Frameworks":   "#1ABC9C",   # mint
    "Other":            "#95A5A6",   # grey
}

SKILL_DOMAINS = {
    # Programming
    "python":     "Programming", "r":            "Programming",
    "sql":        "Programming", "scala":         "Programming",
    "julia":      "Programming", "bash":          "Programming",
    "java":       "Programming", "javascript":    "Programming",
    "typescript": "Programming",
    # ML / DL
    "scikit-learn":  "ML / DL", "tensorflow":  "ML / DL",
    "pytorch":       "ML / DL", "keras":        "ML / DL",
    "jax":           "ML / DL", "xgboost":      "ML / DL",
    "lightgbm":      "ML / DL", "catboost":     "ML / DL",
    "huggingface":   "ML / DL",
    # NLP
    "spacy":    "NLP", "nltk":        "NLP",
    "transformers": "NLP", "bert": "NLP",
    "gpt":      "NLP", "langchain":   "NLP",
    "sentence_transformers": "NLP",
    # Computer Vision
    "opencv":      "Computer Vision", "pillow":    "Computer Vision",
    "yolo":        "Computer Vision", "detectron2":"Computer Vision",
    "clip":        "Computer Vision", "stable_diffusion": "Computer Vision",
    "cuda":        "Computer Vision",
    # Data Processing
    "pandas":  "Data Processing", "numpy":  "Data Processing",
    "polars":  "Data Processing", "dask":   "Data Processing",
    "spark":   "Data Processing", "hadoop": "Data Processing",
    "kafka":   "Data Processing", "flink":  "Data Processing",
    # Visualisation / BI
    "matplotlib": "Visualisation", "seaborn":  "Visualisation",
    "plotly":     "Visualisation", "tableau":  "Visualisation",
    "power_bi":   "Visualisation", "streamlit":"Visualisation",
    "excel":      "Visualisation",
    # Cloud / MLOps / DevOps
    "aws":        "Cloud / MLOps", "gcp":       "Cloud / MLOps",
    "azure":      "Cloud / MLOps", "sagemaker": "Cloud / MLOps",
    "docker":     "Cloud / MLOps", "kubernetes":"Cloud / MLOps",
    "mlflow":     "Cloud / MLOps", "airflow":   "Cloud / MLOps",
    "kubeflow":   "Cloud / MLOps", "wandb":     "Cloud / MLOps",
    "dvc":        "Cloud / MLOps", "ci_cd":     "Cloud / MLOps",
    "terraform":  "Cloud / MLOps", "linux":     "Cloud / MLOps",
    "git":        "Cloud / MLOps", "prefect":   "Cloud / MLOps",
    "prometheus": "Cloud / MLOps",
    # Databases
    "postgresql":     "Databases", "mongodb":       "Databases",
    "redis":          "Databases", "elasticsearch": "Databases",
    "snowflake":      "Databases", "bigquery":      "Databases",
    "sqlite":         "Databases", "databricks":    "Databases",
    # Math / Stats
    "statistics":         "Math / Stats", "probability":    "Math / Stats",
    "linear_algebra":     "Math / Stats", "calculus":       "Math / Stats",
    "bayesian_inference": "Math / Stats", "time_series":    "Math / Stats",
    # Web Frameworks
    "fastapi": "Web Frameworks", "flask":  "Web Frameworks",
    "django":  "Web Frameworks", "nodejs": "Web Frameworks",
    "react":   "Web Frameworks",
    # Other
    "latex": "Other", "dbt": "Other", "jupyter": "Other",
}


def _project_2d(skill_vectors: np.ndarray) -> np.ndarray:
    """
    Project skill vectors to 2D using the first two PCA components.
    We use the components already stored (no re-fitting needed).
    The first two dims of skill_vectors ARE the PC1/PC2 coordinates.
    """
    return skill_vectors[:, :2]


def plot_skill_map(
    analyzer,
    candidate_skills: list[str] | None = None,
    save_path: str | None = None,
    title: str | None = None,
    figsize: tuple = (16, 11),
    show: bool = True,
) -> str | None:
    """
    Generate a 2D skill map in PCA latent space.

    Parameters
    ----------
    analyzer        : SkillAnalyzer instance (already loaded)
    candidate_skills: list of raw skill strings for the candidate overlay
    save_path       : if given, saves PNG to this path
    title           : custom plot title
    figsize         : matplotlib figure size
    show            : whether to call plt.show() (set False in CI/headless)

    Returns
    -------
    save_path if saved, else None
    """
    from normalizer import normalize_skill_list

    skill_names   = analyzer.skill_names
    skill_vectors = analyzer.skill_vectors     # (n_skills, n_components)

    # ── 2D projection ─────────────────────────────────────────────
    coords = _project_2d(skill_vectors)        # (n_skills, 2)

    # ── Domain colours per skill ───────────────────────────────────
    colors = [
        DOMAIN_COLORS.get(SKILL_DOMAINS.get(s, "Other"), DOMAIN_COLORS["Other"])
        for s in skill_names
    ]

    # ── Candidate data ─────────────────────────────────────────────
    candidate_norm = []
    recommendations = []
    candidate_xy = None

    if candidate_skills:
        candidate_norm = normalize_skill_list(candidate_skills, fuzzy=True)
        result         = analyzer.get_insights(candidate_skills)
        recommendations = [r["skill"] for r in result["upskilling"]["recommended_skills"]]

        # Centroid of candidate skill positions
        cand_indices = [analyzer._skill_idx[s] for s in candidate_norm
                        if s in analyzer._skill_idx]
        if cand_indices:
            cand_coords  = coords[cand_indices]
            candidate_xy = cand_coords.mean(axis=0)

    # ── Figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    # ── Background grid ────────────────────────────────────────────
    ax.grid(True, color="#FFFFFF", alpha=0.05, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#666666", labelsize=8)

    # ── Draw all skill points ───────────────────────────────────────
    for i, (skill, (x, y), color) in enumerate(zip(skill_names, coords, colors)):
        is_candidate    = skill in candidate_norm
        is_recommended  = skill in recommendations

        if is_candidate:
            # Candidate's known skills — bright gold ring
            ax.scatter(x, y, s=200, color="#FFD700", zorder=5,
                       edgecolors="#FFF8DC", linewidths=1.5)
            ax.annotate(
                skill, (x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#FFD700",
                zorder=6,
            )
        elif is_recommended:
            # Recommended skills — bright green with pulsing ring
            ax.scatter(x, y, s=160, color="#00FF88", zorder=5,
                       edgecolors="#FFFFFF", linewidths=1.2)
            ax.annotate(
                f"→ {skill}", (x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#00FF88",
                zorder=6,
            )
            # Draw an arrow from candidate centroid to each recommended skill
            if candidate_xy is not None:
                ax.annotate(
                    "", xy=(x, y), xytext=candidate_xy,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#00FF8866",
                        lw=1.2,
                        connectionstyle="arc3,rad=0.1",
                    ),
                    zorder=4,
                )
        else:
            # Regular skill point
            ax.scatter(x, y, s=60, color=color, alpha=0.7, zorder=3,
                       edgecolors="#FFFFFF22", linewidths=0.4)
            ax.annotate(
                skill, (x, y),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6.5, color="#CCCCCC", alpha=0.8,
                zorder=3,
            )

    # ── Candidate centroid star ─────────────────────────────────────
    if candidate_xy is not None:
        ax.scatter(*candidate_xy, s=450, marker="*",
                   color="#FFD700", edgecolors="#FFF8DC",
                   linewidths=1.5, zorder=7, label="You")
        ax.annotate(
            "YOU", candidate_xy,
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, fontweight="bold", color="#FFD700",
            zorder=8,
        )

    # ── Legend ─────────────────────────────────────────────────────
    domain_handles = [
        mpatches.Patch(color=color, label=domain)
        for domain, color in DOMAIN_COLORS.items()
        if any(SKILL_DOMAINS.get(s) == domain for s in skill_names)
    ]
    special_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700",
               markersize=9, label="Your skills"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00FF88",
               markersize=9, label="Recommended"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD700",
               markersize=13, label="Your position"),
    ]
    legend = ax.legend(
        handles=domain_handles + special_handles,
        loc="lower left",
        fontsize=7.5,
        framealpha=0.3,
        facecolor="#111111",
        edgecolor="#333333",
        labelcolor="#CCCCCC",
        ncol=2,
    )

    # ── Title and labels ────────────────────────────────────────────
    plot_title = title or "Skill Intelligence Map — Latent Space (PCA)"
    ax.set_title(plot_title, color="#FFFFFF", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Principal Component 1", color="#666666", fontsize=9)
    ax.set_ylabel("Principal Component 2", color="#666666", fontsize=9)

    if candidate_skills and recommendations:
        subtitle = (
            f"Your skills: {', '.join(candidate_norm[:5])}{'…' if len(candidate_norm) > 5 else ''}   |   "
            f"Recommendations: {', '.join(recommendations)}"
        )
        ax.set_title(plot_title + f"\n{subtitle}",
                     color="#FFFFFF", fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()

    # ── Save / show ─────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✅  Skill map saved → {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass   # headless environment

    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a skill map visualization.")
    parser.add_argument("--skills", type=str, default="",
                        help='Comma-separated skill list, e.g. "tableau,power_bi,excel"')
    parser.add_argument("--out",    type=str, default="skill_map.png",
                        help="Output PNG path (default: skill_map.png)")
    parser.add_argument("--title",  type=str, default=None)
    args = parser.parse_args()

    from inference import SkillAnalyzer
    analyzer = SkillAnalyzer(models_dir=THIS_DIR / "models")

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    plot_skill_map(analyzer, candidate_skills=skills or None,
                   save_path=args.out, title=args.title, show=False)
