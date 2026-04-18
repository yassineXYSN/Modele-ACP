"""
merge_real_datasets.py
======================
Merges all REAL CV/job datasets into a single clean CSV with two columns:
  profile  |  skills

Sources merged:
  1. data/real_cvs.jsonl                     (pre-extracted Kaggle CV dataset)
  2. data/raw/AI_Resume_Screening.csv         (AI resume screening, 1000 rows)
  3. data/raw/Data Science Job Postings & Skills (2024) job_skills.csv (12k rows)

⚠️  Synthetic data (cvs_combined.jsonl) is intentionally EXCLUDED.
⚠️  Only canonical skills (known to the normalizer) are kept — noisy phrases dropped.

Output:
  data/real_merged.csv   — profile, skills  (comma-separated, normalised)

Usage
-----
  python merge_real_datasets.py
  python merge_real_datasets.py --out data/my_output.csv --min-skills 3
"""

import os, sys, re, argparse
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from normalizer import normalize_skill_list, _ALIAS_TO_CANONICAL   # type: ignore

DATA_DIR = ROOT_DIR / "data"
DATA_RAW = ROOT_DIR / "data" / "raw"

# The full set of canonical skill names (used to filter noise)
KNOWN_SKILLS: set[str] = set(_ALIAS_TO_CANONICAL.values())


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE MAPPING
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_MAP: list[tuple[str, str]] = [
    ("nlp",                  "nlp_engineer"),
    ("natural language",     "nlp_engineer"),
    ("computer vision",      "cv_engineer"),
    ("cv engineer",          "cv_engineer"),
    ("image recogni",        "cv_engineer"),
    ("vision engineer",      "cv_engineer"),
    ("mlops",                "mlops_engineer"),
    ("ml ops",               "mlops_engineer"),
    ("platform engineer",    "mlops_engineer"),
    ("infrastructure eng",   "mlops_engineer"),
    ("devops",               "mlops_engineer"),
    ("ml engineer",          "ml_engineer"),
    ("machine learning eng", "ml_engineer"),
    ("ai engineer",          "ml_engineer"),
    ("deep learning",        "ml_engineer"),
    ("ai researcher",        "ai_researcher"),
    ("research scientist",   "ai_researcher"),
    ("research engineer",    "ai_researcher"),
    ("researcher",           "ai_researcher"),
    ("data scientist",       "data_scientist"),
    ("data science",         "data_scientist"),
    ("data engineer",        "data_engineer"),
    ("data engineering",     "data_engineer"),
    ("etl",                  "data_engineer"),
    ("warehouse",            "data_engineer"),
    ("pipeline",             "data_engineer"),
    ("database",             "data_engineer"),
    ("dba",                  "data_engineer"),
    ("data analyst",         "data_analyst"),
    ("analytics",            "data_analyst"),
    ("business analyst",     "data_analyst"),
    ("bi analyst",           "data_analyst"),
    ("reporting",            "data_analyst"),
    ("machine learning",     "ml_engineer"),
    ("software engineer",    "data_engineer"),
    ("full stack",           "data_engineer"),
    ("backend",              "data_engineer"),
]

def map_role(raw: str) -> str:
    s = raw.lower().strip()
    for keyword, profile in _ROLE_MAP:
        if keyword in s:
            return profile
    return "data_scientist"


def _title_from_url(url: str) -> str:
    """Extract job title slug from a LinkedIn URL."""
    m = re.search(r"jobs/view/(.+?)(?:-at-|-\d{10,})", url)
    if m:
        return m.group(1).replace("-", " ")
    slug = url.rstrip("/").split("/")[-1]
    return re.sub(r"-\d{7,}$", "", slug).replace("-", " ")


def filter_to_known(skills: list[str]) -> list[str]:
    """Keep only skills that exist in our canonical vocabulary."""
    return [s for s in skills if s in KNOWN_SKILLS]


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — real_cvs.jsonl  (pre-extracted Kaggle Curriculum Vitae dataset)
# ─────────────────────────────────────────────────────────────────────────────

def load_real_cvs_jsonl(path: Path, min_skills: int) -> list[dict]:
    import json
    print(f"\n📄  Loading: {path.name}  (pre-extracted Kaggle CVs)")
    if not path.exists():
        print(f"   ⚠️  File not found: {path}")
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            skills = normalize_skill_list(rec.get("skills_raw", []), fuzzy=True)
            skills = filter_to_known(skills)
            skills = list(dict.fromkeys(skills))          # dedup
            if len(skills) < min_skills:
                continue
            profile = rec.get("profile", "data_scientist")
            records.append({"profile": profile, "skills": skills})

    print(f"   ✅  Valid rows: {len(records)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — AI_Resume_Screening.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_ai_resume_screening(path: Path, min_skills: int) -> list[dict]:
    print(f"\n📄  Loading: {path.name}")
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

    print(f"   Columns : {list(df.columns)}")
    print(f"   Rows    : {len(df)}")

    SKILL_COLS = ["Skills", "skills", "Skill", "Technologies", "tech_stack"]
    ROLE_COLS  = ["Job Role", "job_role", "Role", "Title", "Category", "Profile"]

    skill_col = next((c for c in SKILL_COLS if c in df.columns), None)
    role_col  = next((c for c in ROLE_COLS  if c in df.columns), None)

    if skill_col is None:
        print("   ❌  No skills column found. Skipping.")
        return []

    print(f"   Skills col : '{skill_col}' | Role col : '{role_col or 'none'}'")

    records = []
    for _, row in df.iterrows():
        raw_skills = str(row[skill_col]) if pd.notna(row.get(skill_col)) else ""
        raw_role   = str(row[role_col])  if role_col and pd.notna(row.get(role_col)) else ""

        raw_list = [s.strip() for s in re.split(r"[,;|]", raw_skills) if s.strip()]
        skills   = normalize_skill_list(raw_list, fuzzy=True)
        skills   = filter_to_known(skills)
        skills   = list(dict.fromkeys(skills))

        if len(skills) < min_skills:
            continue

        profile = map_role(raw_role) if raw_role else "data_scientist"
        records.append({"profile": profile, "skills": skills})

    print(f"   ✅  Valid rows: {len(records)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — Data Science Job Postings job_skills.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_job_postings(path: Path, min_skills: int) -> list[dict]:
    print(f"\n📄  Loading: {path.name}")
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

    print(f"   Columns : {list(df.columns)}")
    print(f"   Rows    : {len(df)}")

    LINK_COLS  = ["job_link", "url", "link", "job_url"]
    SKILL_COLS = ["job_skills", "skills", "Skills", "required_skills"]

    link_col  = next((c for c in LINK_COLS  if c in df.columns), None)
    skill_col = next((c for c in SKILL_COLS if c in df.columns), None)

    if skill_col is None:
        print("   ❌  No skills column found. Skipping.")
        return []

    print(f"   Skills col : '{skill_col}' | Link col : '{link_col or 'none'}'")

    records = []
    for _, row in df.iterrows():
        raw_skills = str(row[skill_col]) if pd.notna(row.get(skill_col)) else ""
        raw_link   = str(row[link_col])  if link_col and pd.notna(row.get(link_col)) else ""

        title   = _title_from_url(raw_link) if raw_link else ""
        profile = map_role(title) if title else "data_scientist"

        raw_list = [s.strip() for s in re.split(r"[,;|]", raw_skills) if s.strip()]
        skills   = normalize_skill_list(raw_list, fuzzy=True)
        skills   = filter_to_known(skills)             # ← drop unknown/noisy phrases
        skills   = list(dict.fromkeys(skills))

        if len(skills) < min_skills:
            continue

        records.append({"profile": profile, "skills": skills})

    print(f"   ✅  Valid rows: {len(records)}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(df: pd.DataFrame, skills_lists: list[list[str]]) -> None:
    print(f"\n{'═'*55}")
    print(f"  Final Dataset Statistics")
    print(f"{'═'*55}")
    print(f"  Total rows     : {len(df)}")

    print(f"\n  Profile distribution:")
    prof_counts = df["profile"].value_counts()
    for prof, cnt in prof_counts.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {prof:<30} {cnt:>5}  ({pct:.1f}%)  {bar}")

    all_skills = [s for row in skills_lists for s in row]
    freq       = Counter(all_skills)
    print(f"\n  Unique skills  : {len(freq)}")
    print(f"  Avg skills/row : {len(all_skills)/len(df):.1f}")
    print(f"\n  Top 25 skills:")
    for skill, count in freq.most_common(25):
        pct = count / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f"  {skill:<30} {pct:>5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(out_path: str, min_skills: int) -> None:
    print("═" * 55)
    print("  🔀  Real Dataset Merger")
    print("  (synthetic data excluded)")
    print("═" * 55)

    all_records: list[dict] = []
    source_counts = {}

    # Source 1 — Kaggle CVs (from pre-extracted JSONL)
    recs = load_real_cvs_jsonl(DATA_DIR / "real_cvs.jsonl", min_skills)
    source_counts["Curriculum Vitae (Kaggle)"] = len(recs)
    all_records.extend(recs)

    # Source 2 — AI Resume Screening
    ai_path = DATA_RAW / "AI_Resume_Screening.csv"
    if ai_path.exists():
        recs = load_ai_resume_screening(ai_path, min_skills)
        source_counts["AI_Resume_Screening"] = len(recs)
        all_records.extend(recs)
    else:
        print(f"\n⚠️  Not found: {ai_path.name}")

    # Source 3 — Job Postings (try exact name then glob)
    job_path = DATA_RAW / "Data Science Job Postings & Skills (2024) job_skills.csv"
    if not job_path.exists():
        matches = list(DATA_RAW.glob("*job_skills*"))
        job_path = matches[0] if matches else None

    if job_path:
        recs = load_job_postings(job_path, min_skills)
        source_counts["Job Postings (LinkedIn 2024)"] = len(recs)
        all_records.extend(recs)
    else:
        print("\n⚠️  job_skills.csv not found")

    if not all_records:
        print("\n❌  No records loaded. Check dataset paths.")
        sys.exit(1)

    # Source breakdown
    print(f"\n{'─'*45}")
    print(f"  Source breakdown:")
    for src, cnt in source_counts.items():
        print(f"  {src:<35} {cnt:>5} rows")
    print(f"  {'TOTAL':<35} {len(all_records):>5} rows")

    # Build output DataFrame
    skills_lists = [r["skills"] for r in all_records]
    df = pd.DataFrame({
        "profile": [r["profile"] for r in all_records],
        "skills":  [", ".join(r["skills"]) for r in all_records],
    })

    print_stats(df, skills_lists)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n💾  Saved → {out_path}")
    print(f"   Columns : profile, skills")
    print(f"   Rows    : {len(df)}")
    print(f"\n✅  Done!")
    print(f"   To retrain the model on this data, convert to JSONL first:")
    print(f"   python train_pipeline.py --data {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Merge all real CV/job datasets into a clean profile+skills CSV."
    )
    ap.add_argument("--out",        default="data/real_merged.csv",
                    help="Output CSV path (default: data/real_merged.csv)")
    ap.add_argument("--min-skills", type=int, default=3,
                    help="Minimum skills per row to keep it (default: 3)")
    args = ap.parse_args()
    main(out_path=args.out, min_skills=args.min_skills)
