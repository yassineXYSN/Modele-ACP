"""
test_manual_skills.py
=====================
Interactive CLI form — type your skills and get AI-powered recommendations.

Usage
-----
  python test_manual_skills.py

Features
--------
  • Type skills one at a time — auto-normalised as you go
  • Fuzzy matching: type "Pytoch" and it corrects to "pytorch"
  • Unknown skills flagged so you can retype them
  • Full SkillAnalyzer output: upskilling + importance + liaison
  • Compare two skill sets (e.g. yourself vs a job description)
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives at project root (Modele-ACP/)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = THIS_DIR   # already at root
sys.path.insert(0, ROOT_DIR)

# ── Optional visualization ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # works headless AND with display
    from visualize import plot_skill_map
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from inference import SkillAnalyzer
except ImportError as e:
    sys.exit(f"❌  Could not import SkillAnalyzer: {e}")

from normalizer import normalize_skill, normalize_skill_list

MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def _color(text, code):
    return f"{code}{text}{RESET}"

def _banner(text: str) -> None:
    w = 65
    print(f"\n{BOLD}{'═' * w}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'═' * w}{RESET}\n")

def _section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{'─' * 65}")

def _prompt(text: str) -> str:
    return input(f"  {_color('▶', CYAN)} {text}").strip()

# ─────────────────────────────────────────────────────────────────────────────
# SKILL INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────

def _collect_skills(analyzer: SkillAnalyzer, label: str = "your") -> list[str]:
    """
    Interactive skill collection loop.
    Returns a list of canonical skill names known to the model.
    """
    _section(f"📝  Enter {label} skills")
    print(f"  {DIM}Type one skill per line. Press Enter on an empty line when done.{RESET}")
    print(f"  {DIM}Tip: typos are auto-corrected (e.g. 'Pytoch' → pytorch){RESET}\n")

    collected: list[str]  = []   # canonical names
    raw_entries: list[str] = []  # what the user typed
    unknown: list[str]    = []   # couldn't be normalised to a known vocab term

    vocab = set(analyzer.skill_names)   # known skills in the model

    idx = 1
    while True:
        raw = _prompt(f"Skill {idx:>2}: ")
        if not raw:
            if not collected:
                print(f"  {_color('⚠️  Please enter at least one skill.', YELLOW)}")
                continue
            break

        canonical = normalize_skill(raw, fuzzy=True)

        if canonical is None:
            # Soft skill or stop-word — silently skip
            print(f"  {_color('  ↳ Soft skill / noise — skipped', DIM)}")
            continue

        # Check if it's in the model vocabulary
        if canonical in vocab:
            if canonical in collected:
                already_msg = f'  \u21b3 Already added as "{canonical}"'
                print(f"  {_color(already_msg, DIM)}")
            else:
                collected.append(canonical)
                raw_entries.append(raw)
                arrow_msg = f"  (\u2192 {canonical})"
                indicator = "" if canonical == raw.lower() else f"  {_color(arrow_msg, GREEN)}"
                print(f"  {_color('  ✅', GREEN)}{indicator}")
                idx += 1
        else:
            # Normalised but not in model vocab — keep it but flag it
            if canonical not in collected:
                collected.append(canonical)
                raw_entries.append(raw)
                unknown.append(canonical)
                warn_msg = f'  \u26a0\ufe0f  Normalised to "{canonical}" but not in model vocabulary \u2014 kept, may not affect recommendations'
                print(f"  {_color(warn_msg, YELLOW)}")
                idx += 1
            else:
                print(f"  {_color('  ↳ Already added', DIM)}")

    # Summary
    known_count   = len([s for s in collected if s not in unknown])
    unknown_count = len(unknown)
    print(f"\n  📋  Collected {len(collected)} skills  "
          f"({_color(f'{known_count} recognised', GREEN)}"
          + (f", {_color(f'{unknown_count} unknown', YELLOW)}" if unknown_count else "")
          + ")")
    print(f"  Skills: {_color(', '.join(collected), CYAN)}")

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def display_upskilling(result: dict) -> None:
    recs = result["upskilling"]["recommended_skills"]
    _section("📈  Skills You Should Learn Next")

    if not recs:
        print(f"  {_color('You already cover most of the high-value skills! 🎉', GREEN)}")
        return

    print(f"  These are the skills closest to your current profile:\n")
    for i, r in enumerate(recs, 1):
        bar_len = max(1, int(r["proximity_score"] * 28))
        bar     = _color("▓" * bar_len, GREEN)
        score   = f"{r['proximity_score']:.3f}"
        print(f"  {BOLD}{i}.{RESET} {r['skill']:<25}  {bar:<30}  {DIM}{score}{RESET}")

        hint = _upskill_hints().get(r["skill"])
        if hint:
            print(f"     {DIM}└─ {hint}{RESET}")


def display_importance(result: dict, user_skills: list[str]) -> None:
    _section("🏆  Most Discriminating Skills in This Domain")
    print(f"  {DIM}High score = this skill separates different job profiles the most{RESET}\n")

    user_set = set(user_skills)
    for r in result["skill_importance"]["ranked_skills"][:10]:
        owned = r["skill"] in user_set
        marker = _color("⭐ you", GREEN) if owned else "     "
        name   = _color(r["skill"], BOLD) if owned else r["skill"]
        print(f"  {marker}  {name:<25}  {r['importance_score']:.4f}")


def display_liaison(analyzer: SkillAnalyzer, skills: list[str]) -> None:
    _section("🔗  Skill Liaison (how related are your skills to each other?)")

    if len(skills) < 2:
        print("  (Need at least 2 skills)")
        return

    vocab = set(analyzer.skill_names)
    known = [s for s in skills if s in vocab]

    if len(known) < 2:
        print("  (Not enough recognised skills for liaison analysis)")
        return

    # Build all pairs up to 6 skills
    pairs = [(known[i], known[j])
             for i in range(min(6, len(known)))
             for j in range(i+1, min(6, len(known)))]

    print(f"  {'Skill A':<22}  {'Skill B':<22}  {'Similarity':>10}  Strength")
    print("  " + "─" * 62)
    for a, b in pairs[:8]:
        r = analyzer.skill_liaison(a, b)
        if "error" in r:
            continue
        sim = r["cosine_similarity"]
        if sim > 0.6:
            strength = _color("████  Strong", GREEN)
        elif sim > 0.35:
            strength = _color("██░░  Moderate", YELLOW)
        else:
            strength = _color("░░░░  Weak", DIM)
        print(f"  {a:<22}  {b:<22}  {sim:>10.3f}  {strength}")


def display_gap_analysis(analyzer: SkillAnalyzer,
                          my_skills: list[str],
                          job_skills: list[str]) -> None:
    """Compare two skill sets and show the gap."""
    _section("📊  Gap Analysis  (Your Skills vs Job Requirements)")

    my_set  = set(my_skills)
    job_set = set(job_skills)

    have     = sorted(my_set & job_set)
    missing  = sorted(job_set - my_set)
    extra    = sorted(my_set - job_set)

    print(f"  {_color('✅ Skills you have that the job wants:', GREEN)}")
    if have:
        for s in have:
            print(f"     • {s}")
    else:
        print(f"     {DIM}(none){RESET}")

    print(f"\n  {_color('❌ Skills the job wants that you are missing:', RED)}")
    if missing:
        for s in missing:
            r = analyzer.skill_liaison(missing[0], s) if missing else {}
            print(f"     • {s}")
    else:
        print(f"     {_color('(none — great match! 🎉)', GREEN)}")

    print(f"\n  {_color('➕ Extra skills you have (bonus):', CYAN)}")
    if extra:
        for s in list(extra)[:10]:
            print(f"     • {s}")
    else:
        print(f"     {DIM}(none){RESET}")

    pct = int(100 * len(have) / len(job_set)) if job_set else 0
    bar = _color("█" * (pct // 5), GREEN) + _color("░" * (20 - pct // 5), DIM)
    print(f"\n  Match score: [{bar}] {pct}%")


def _upskill_hints() -> dict[str, str]:
    return {
        "pytorch":      "Great for deep learning research. Start with fast.ai.",
        "tensorflow":   "Used in production. Try Google's ML crash course.",
        "docker":       "Containerise your models. Essential for deployment.",
        "mlflow":       "Track experiments and manage model versions.",
        "airflow":      "Orchestrate ML pipelines. Key in data engineering.",
        "kubernetes":   "Deploy at scale. Pair with docker knowledge.",
        "spark":        "Handle big data. Great for large-scale feature engineering.",
        "dbt":          "Transform data in the warehouse. Very in-demand.",
        "fastapi":      "Build ML APIs quickly. Built on async Python.",
        "scikit-learn": "The ML Swiss army knife. Core for classical ML.",
        "xgboost":      "Top Kaggle model. Outperforms RF on tabular data.",
        "lightgbm":     "Faster than XGBoost. Great for large datasets.",
        "huggingface":  "Access 100k+ pretrained NLP/CV models instantly.",
        "bert":         "Foundation for modern NLP. Fine-tuning is key.",
        "langchain":    "Build LLM-powered apps. Very hot right now.",
        "wandb":        "Experiment tracking. Pairs well with PyTorch.",
        "streamlit":    "Turn your model into a web app in 10 lines of code.",
        "excel":        "Still essential for data analysts and stakeholders.",
        "power_bi":     "Business-facing dashboards. Pair with SQL.",
        "tableau":      "Industry-standard visualisation tool.",
        "r":            "Statistical computing. Strong in research/academia.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _offer_skill_map(analyzer, skills: list[str]) -> None:
    """Ask the user if they want a skill map PNG generated."""
    if not _VIZ_AVAILABLE:
        return
    print()
    choice = _prompt("Generate a skill map PNG? (y/n): ").lower()
    if choice != "y":
        return

    default_name = "skill_map_" + "_".join(skills[:2]) + ".png"
    path_input = _prompt(f"Save as [{default_name}]: ").strip()
    save_path  = path_input if path_input else default_name

    # Make sure directory exists
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    plot_skill_map(analyzer, candidate_skills=skills,
                   save_path=save_path, show=False)
    print(f"  {_color('Skill map saved to: ' + save_path, GREEN)}")


# ─────────────────────────────────────────────────────────────────────────────
# MENU
# ─────────────────────────────────────────────────────────────────────────────

def main_menu(analyzer: SkillAnalyzer) -> None:
    while True:
        _banner("🧠  Skill Intelligence — Manual Form")
        print("  What would you like to do?\n")
        print(f"  {BOLD}1.{RESET}  Enter my skills → get upskilling recommendations")
        print(f"  {BOLD}2.{RESET}  Compare my skills to a job description")
        print(f"  {BOLD}3.{RESET}  Explore skill relationships (liaison)")
        print(f"  {BOLD}4.{RESET}  Browse all skills the model knows")
        print(f"  {BOLD}q.{RESET}  Quit\n")

        choice = _prompt("Choice: ").lower()

        if choice == "1":
            mode_upskilling(analyzer)
        elif choice == "2":
            mode_gap_analysis(analyzer)
        elif choice == "3":
            mode_liaison(analyzer)
        elif choice == "4":
            mode_browse(analyzer)
        elif choice in ("q", "quit", "exit"):
            print(f"\n  {_color('Goodbye! 👋', GREEN)}\n")
            break
        else:
            print(f"  {_color('Invalid choice — please try again.', YELLOW)}")


def mode_upskilling(analyzer: SkillAnalyzer) -> None:
    skills = _collect_skills(analyzer, label="your")
    if not skills:
        return

    _section("🧠  Running Skill Analysis …")
    result = analyzer.get_insights(skills)

    display_upskilling(result)
    display_importance(result, skills)
    display_liaison(analyzer, skills)

    print(f"\n  {DIM}Profile vector ({len(result['profile_vector'])} dims): "
          f"[{', '.join(f'{v:.3f}' for v in result['profile_vector'][:5])} …]{RESET}")

    _offer_skill_map(analyzer, skills)
    input(f"\n  {DIM}Press Enter to return to the menu …{RESET}")


def mode_gap_analysis(analyzer: SkillAnalyzer) -> None:
    print(f"\n  {DIM}Step 1: Enter YOUR current skills{RESET}")
    my_skills = _collect_skills(analyzer, label="your")

    print(f"\n  {DIM}Step 2: Enter the REQUIRED SKILLS from the job posting{RESET}")
    job_skills = _collect_skills(analyzer, label="the job's required")

    if not my_skills or not job_skills:
        return

    display_gap_analysis(analyzer, my_skills, job_skills)

    # Also show upskilling from MY perspective
    result = analyzer.get_insights(my_skills)
    display_upskilling(result)

    input(f"\n  {DIM}Press Enter to return to the menu …{RESET}")


def mode_liaison(analyzer: SkillAnalyzer) -> None:
    _section("🔗  Skill Relationship Explorer")
    print(f"  Enter two skills to see how related they are.\n")

    vocab = set(analyzer.skill_names)
    while True:
        a_raw = _prompt("Skill A (or 'back'): ")
        if a_raw.lower() in ("back", "b", ""):
            return
        b_raw = _prompt("Skill B: ")
        if not b_raw:
            return

        a = normalize_skill(a_raw, fuzzy=True)
        b = normalize_skill(b_raw, fuzzy=True)

        if not a:
            print(f"  {_color('Could not recognise: ' + a_raw, RED)}")
            continue
        if not b:
            print(f"  {_color('Could not recognise: ' + b_raw, RED)}")
            continue

        r = analyzer.skill_liaison(a, b)
        if "error" in r:
            print(f"  {_color(r['error'], RED)}")
        else:
            sim = r["cosine_similarity"]
            bar = _color("█" * max(0, int(sim * 30)), GREEN if sim > 0.5 else YELLOW)
            print(f"\n  {a:<20} ↔  {b:<20}")
            print(f"  Cosine similarity: {BOLD}{sim:.4f}{RESET}  [{bar}]")
            print(f"  {_color(r['interpretation'], CYAN)}\n")

        again = _prompt("Try another pair? (y/n): ").lower()
        if again != "y":
            return


def mode_browse(analyzer: SkillAnalyzer) -> None:
    _section("📚  All Skills in the Model")
    skills = sorted(analyzer.skill_names)

    # Group by rough category
    categories = {
        "🐍 Programming":       ["python", "r", "sql", "scala", "julia", "bash", "java",
                                  "javascript", "typescript", "go", "rust"],
        "🌐 Web Frameworks":    ["react", "vue", "angular", "nodejs", "nextjs",
                                  "fastapi", "flask", "django"],
        "🤖 ML / DL":          ["scikit-learn", "tensorflow", "pytorch", "keras", "jax",
                                  "xgboost", "lightgbm", "catboost", "huggingface"],
        "💬 NLP":              ["spacy", "nltk", "transformers", "bert", "gpt", "langchain",
                                  "sentence_transformers"],
        "👁️ Computer Vision":  ["opencv", "pillow", "yolo", "detectron2", "clip",
                                  "stable_diffusion"],
        "📊 Data Processing":  ["pandas", "numpy", "polars", "dask", "spark",
                                  "hadoop", "kafka", "flink"],
        "📈 Visualisation":    ["matplotlib", "seaborn", "plotly", "tableau",
                                  "power_bi", "streamlit", "excel"],
        "☁️ Cloud / MLOps":    ["aws", "gcp", "azure", "sagemaker", "docker",
                                  "kubernetes", "mlflow", "airflow", "wandb", "dvc"],
        "🗄️ Databases":        ["postgresql", "mongodb", "redis", "elasticsearch",
                                  "snowflake", "bigquery", "sqlite"],
        "📐 Math / Stats":     ["statistics", "probability", "linear_algebra",
                                  "calculus", "bayesian_inference", "time_series"],
    }

    categorised = set()
    for cat_name, cat_skills in categories.items():
        in_model = [s for s in cat_skills if s in skills]
        if in_model:
            print(f"\n  {BOLD}{cat_name}{RESET}")
            print(f"  {', '.join(_color(s, CYAN) for s in in_model)}")
            categorised.update(in_model)

    other = [s for s in skills if s not in categorised]
    if other:
        print(f"\n  {BOLD}Other{RESET}")
        print(f"  {', '.join(_color(s, DIM) for s in other)}")

    print(f"\n  {DIM}Total: {len(skills)} skills{RESET}")
    input(f"\n  {DIM}Press Enter to return to the menu …{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _banner("🧠  Skill Intelligence — Loading Model")
    try:
        analyzer = SkillAnalyzer(models_dir=MODELS_DIR)
    except Exception as e:
        sys.exit(f"❌  Failed to load model: {e}")

    main_menu(analyzer)
