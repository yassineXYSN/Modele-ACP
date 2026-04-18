"""
normalizer.py — Canonical Skill Normalizer
==========================================
Handles the full mess that LLM-parsed CVs produce:
  • Casing  : "Python", "PYTHON", "python"  → python
  • Aliases : "ReactJS", "React.js", "react" → react
  • Versions: "Python 3.9", "Node 18"        → python / nodejs
  • Typos   : "Postgers", "Pytoch"           → postgresql / pytorch  (fuzzy)
  • Noise   : "good communication skills"    → [dropped]

Usage
-----
from normalizer import normalize_skill, normalize_skill_list

skills = normalize_skill_list(["React.js", "Python 3.9", "Sci-kit learn",
                               "NodeJS", "TF", "k8s", "good communication"])
# → ['react', 'python', 'scikit-learn', 'nodejs', 'tensorflow', 'kubernetes']
"""

import re
from difflib import get_close_matches

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CANONICAL NAME  →  all accepted aliases (lowercase, stripped)
# ──────────────────────────────────────────────────────────────────────────────
_CANONICAL_MAP: dict[str, list[str]] = {

    # ── Languages ────────────────────────────────────────────────────────────
    "python":       ["python", "py", "python3", "python2", "python 3", "python 2",
                     "python programming", "python language"],
    "r":            ["r", "r language", "r programming", "rlang"],
    "sql":          ["sql", "mysql", "t-sql", "tsql", "ms sql", "ms-sql", "mssql",
                     "pl/sql", "plsql", "sql server", "ansi sql", "structured query language"],
    "scala":        ["scala"],
    "julia":        ["julia", "julialang"],
    "bash":         ["bash", "shell", "shell scripting", "bash scripting", "unix shell",
                     "sh", "zsh"],
    "java":         ["java", "java se", "java ee", "core java"],
    "javascript":   ["javascript", "js", "es6", "es7", "es8", "ecmascript",
                     "es2015", "es2016", "es2017", "es2018", "es2019", "es2020", "es2021",
                     "vanilla js", "vanilla javascript"],
    "typescript":   ["typescript", "ts"],
    "cpp":          ["c++", "cpp", "c plus plus"],
    "csharp":       ["c#", "csharp", "c sharp", ".net"],
    "rust":         ["rust", "rust lang", "rustlang"],
    "go":           ["go", "golang"],

    # ── Web Frameworks ───────────────────────────────────────────────────────
    "react":        ["react", "reactjs", "react.js", "react js",
                     "react native", "reactnative"],
    "vue":          ["vue", "vuejs", "vue.js", "vue js", "vue 3", "vue 2"],
    "angular":      ["angular", "angularjs", "angular.js", "angular js",
                     "angular 2", "angular 14", "angular 15"],
    "nodejs":       ["nodejs", "node.js", "node js", "node", "expressjs",
                     "express.js", "express js", "express"],
    "nextjs":       ["nextjs", "next.js", "next js"],
    "fastapi":      ["fastapi", "fast api", "fast-api"],
    "flask":        ["flask", "flask api"],
    "django":       ["django", "django rest framework", "drf", "django rf"],

    # ── ML / DL Frameworks ───────────────────────────────────────────────────
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn", "sk-learn",
                     "scikitlearn", "sci-kit learn", "scikit_learn", "sci kit learn"],
    "tensorflow":   ["tensorflow", "tf", "tensorflow 2", "tf2", "tensorflow2",
                     "tensorflow/keras", "tf/keras"],
    "pytorch":      ["pytorch", "torch", "pyTorch", "py torch", "pytorch lightning",
                     "pytoch", "pytoch"],   # common typo included
    "keras":        ["keras"],
    "jax":          ["jax", "google jax"],
    "xgboost":      ["xgboost", "xgb", "extreme gradient boosting"],
    "lightgbm":     ["lightgbm", "lgbm", "light gbm", "light gradient boosting"],
    "catboost":     ["catboost", "cat boost"],
    "huggingface":  ["huggingface", "hugging face", "hf", "transformers hub",
                     "huggingface hub", "hugging-face"],

    # ── NLP ──────────────────────────────────────────────────────────────────
    "spacy":        ["spacy", "spaCy", "spacy nlp"],
    "nltk":         ["nltk", "natural language toolkit"],
    "transformers": ["transformers", "huggingface transformers",
                     "hf transformers", "transformer models"],
    "bert":         ["bert", "google bert", "bert model",
                     "roberta", "distilbert", "albert", "xlnet"],
    "gpt":          ["gpt", "gpt-2", "gpt-3", "gpt-4", "gpt4", "gpt3", "openai gpt",
                     "chatgpt", "llm", "large language model"],
    "langchain":    ["langchain", "lang chain", "langchain framework"],
    "sentence_transformers": ["sentence-transformers", "sentence transformers",
                              "sbert", "sentence bert"],

    # ── Computer Vision ──────────────────────────────────────────────────────
    "opencv":       ["opencv", "cv2", "open cv", "open-cv"],
    "pillow":       ["pillow", "pil", "python imaging library"],
    "yolo":         ["yolo", "yolov5", "yolov7", "yolov8", "you only look once",
                     "ultralytics yolo"],
    "detectron2":   ["detectron2", "detectron 2", "facebook detectron"],
    "clip":         ["clip", "openai clip"],
    "stable_diffusion": ["stable diffusion", "stable-diffusion", "stablediffusion",
                         "sd", "sdxl"],

    # ── Data Processing ──────────────────────────────────────────────────────
    "pandas":       ["pandas", "pd", "pandas dataframe"],
    "numpy":        ["numpy", "np", "numerical python"],
    "polars":       ["polars"],
    "dask":         ["dask", "dask dataframe"],
    "spark":        ["spark", "apache spark", "pyspark", "spark sql",
                     "databricks spark"],
    "hadoop":       ["hadoop", "apache hadoop", "hdfs", "mapreduce"],
    "kafka":        ["kafka", "apache kafka"],
    "flink":        ["flink", "apache flink"],

    # ── Visualization ────────────────────────────────────────────────────────
    "matplotlib":   ["matplotlib", "pyplot", "plt"],
    "seaborn":      ["seaborn", "sns"],
    "plotly":       ["plotly", "plotly express", "dash"],
    "tableau":      ["tableau", "tableau desktop", "tableau public"],
    "power_bi":     ["power bi", "powerbi", "power-bi", "microsoft power bi",
                     "ms power bi"],
    "streamlit":    ["streamlit", "streamlit app"],
    "excel":        ["excel", "microsoft excel", "ms excel", "advanced excel",
                     "vba", "excel vba"],

    # ── Databases ────────────────────────────────────────────────────────────
    "postgresql":   ["postgresql", "postgres", "psql", "postgre sql", "postgers",
                     "postgresql db"],
    "mongodb":      ["mongodb", "mongo", "mongo db"],
    "redis":        ["redis", "redis cache"],
    "elasticsearch":["elasticsearch", "elastic search", "elastic", "opensearch"],
    "snowflake":    ["snowflake", "snowflake db", "snowflake data warehouse"],
    "bigquery":     ["bigquery", "google bigquery", "bq"],
    "sqlite":       ["sqlite", "sqlite3", "sqlite db"],
    "databricks":   ["databricks", "azure databricks"],

    # ── Cloud ────────────────────────────────────────────────────────────────
    "aws":          ["aws", "amazon web services", "amazon aws"],
    "sagemaker":    ["sagemaker", "amazon sagemaker", "aws sagemaker"],
    "s3":           ["s3", "amazon s3", "aws s3"],
    "lambda":       ["lambda", "aws lambda", "amazon lambda"],
    "glue":         ["glue", "aws glue"],
    "ec2":          ["ec2", "amazon ec2", "aws ec2"],
    "gcp":          ["gcp", "google cloud", "google cloud platform"],
    "vertex_ai":    ["vertex ai", "vertex_ai", "google vertex", "google vertex ai"],
    "dataflow":     ["dataflow", "google dataflow"],
    "azure":        ["azure", "microsoft azure"],
    "azure_ml":     ["azure ml", "azure machine learning", "azureml"],

    # ── MLOps / DevOps ───────────────────────────────────────────────────────
    "mlflow":       ["mlflow", "ml flow"],
    "kubeflow":     ["kubeflow", "kube flow"],
    "airflow":      ["airflow", "apache airflow"],
    "docker":       ["docker", "dockerfile", "docker container", "docker image"],
    "kubernetes":   ["kubernetes", "k8s", "kube"],
    "dvc":          ["dvc", "data version control"],
    "wandb":        ["wandb", "weights & biases", "weights and biases"],
    "prefect":      ["prefect", "prefect flow"],
    "terraform":    ["terraform", "hashicorp terraform"],
    "ci_cd":        ["ci/cd", "ci cd", "cicd", "continuous integration",
                     "continuous deployment", "continuous delivery",
                     "github actions", "gitlab ci", "jenkins"],
    "linux":        ["linux", "ubuntu", "debian", "centos", "unix",
                     "linux administration"],
    "git":          ["git", "version control"],
    "github":       ["github", "git hub"],
    "gitlab":       ["gitlab", "git lab"],

    # ── Math / Stats ─────────────────────────────────────────────────────────
    "statistics":   ["statistics", "statistical analysis", "stats",
                     "descriptive statistics", "inferential statistics"],
    "probability":  ["probability", "probability theory", "stochastic"],
    "linear_algebra": ["linear algebra", "matrix operations", "linear_algebra"],
    "calculus":     ["calculus", "differential calculus", "integral calculus",
                     "multivariable calculus"],
    "bayesian_inference": ["bayesian", "bayesian inference", "bayesian statistics",
                           "bayesian_inference", "bayes"],
    "time_series":  ["time series", "time-series", "time_series", "arima",
                     "lstm time series", "forecasting"],

    # ── Misc ─────────────────────────────────────────────────────────────────
    "jupyter":      ["jupyter", "jupyter notebook", "jupyter lab", "jupyterlab",
                     "ipython notebook", "ipynb"],
    "latex":        ["latex", "latex typesetting"],
    "dbt":          ["dbt", "data build tool"],
}

# ── Soft-skill stop-words — always dropped ───────────────────────────────────
_SOFT_SKILL_STOPWORDS: set[str] = {
    "communication", "teamwork", "leadership", "proactive", "dynamic", "punctual",
    "rigorous", "autonomous", "motivated", "creative", "flexible", "organised",
    "organized", "hardworking", "detail oriented", "detail-oriented", "self starter",
    "self-starter", "fast learner", "quick learner", "team player", "team spirit",
    "interpersonal skills", "problem solving", "problem-solving", "analytical",
    "adaptable", "agile mindset", "results oriented", "results-driven",
    "good communication", "good communication skills", "strong communication",
    "excellent communication", "written communication", "verbal communication",
}

# ── Phrases that are too generic to keep ─────────────────────────────────────
_GENERIC_STOPWORDS: set[str] = {
    "programming", "development", "software", "technology", "engineering",
    "data", "machine learning", "artificial intelligence", "ai", "ml",  # too broad
    "computer science", "it", "information technology",
    "microsoft office", "office suite", "pack office",  # umbrella terms
    "database", "web development", "backend", "frontend",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  BUILD REVERSE LOOKUP  alias → canonical
# ──────────────────────────────────────────────────────────────────────────────
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _CANONICAL_MAP.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower().strip()] = canonical
    # also map canonical to itself
    _ALIAS_TO_CANONICAL[canonical.lower().strip()] = canonical

# Pre-built sorted list of canonical names for fuzzy matching
_ALL_CANONICALS = sorted(_CANONICAL_MAP.keys())
_ALL_ALIASES    = sorted(_ALIAS_TO_CANONICAL.keys())


# ──────────────────────────────────────────────────────────────────────────────
# 3.  NORMALISATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def _clean_raw(skill: str) -> str:
    """Step 1 — mechanical cleaning before lookup."""
    s = skill.lower().strip()
    # Remove trailing/leading punctuation
    s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
    # Strip version numbers like "python 3.9", "node 18", "vue 3"
    s = re.sub(r"\s*v?\d+(\.\d+)*$", "", s).strip()
    # Normalise separators: "scikit_learn", "scikit-learn", "scikit learn" → consistent
    # (keep as-is for now — handled by alias map)
    return s


def normalize_skill(skill: str, fuzzy: bool = True) -> str | None:
    """
    Normalise a single raw skill string to its canonical form.

    Parameters
    ----------
    skill  : raw string from LLM parser
    fuzzy  : if True, fall back to fuzzy matching when exact lookup fails

    Returns
    -------
    canonical skill name (str) or None if the skill should be dropped.
    """
    if not skill or not skill.strip():
        return None

    cleaned = _clean_raw(skill)

    # Drop soft skills and generic noise
    if cleaned in _SOFT_SKILL_STOPWORDS or cleaned in _GENERIC_STOPWORDS:
        return None
    if any(cleaned == s for s in _SOFT_SKILL_STOPWORDS):
        return None

    # ── Exact lookup ─────────────────────────────────────────────────────────
    if cleaned in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[cleaned]

    # ── Try with underscores / hyphens replaced ───────────────────────────────
    for sep_replaced in [cleaned.replace("-", " "), cleaned.replace("_", " "),
                          cleaned.replace(" ", "_"), cleaned.replace(" ", "-")]:
        if sep_replaced in _ALIAS_TO_CANONICAL:
            return _ALIAS_TO_CANONICAL[sep_replaced]

    # ── Fuzzy fallback (only for longer strings to avoid false positives) ────
    if fuzzy and len(cleaned) >= 4:
        matches = get_close_matches(cleaned, _ALL_ALIASES, n=1, cutoff=0.82)
        if matches:
            return _ALIAS_TO_CANONICAL[matches[0]]

    # ── Unknown but valid-looking skill: keep as-is (lowercased, underscored) ─
    # Replace spaces → underscores so the vectorizer treats it as one token
    normalized = re.sub(r"[\s\-]+", "_", cleaned)
    if len(normalized) >= 2:
        return normalized

    return None


def normalize_skill_list(skills: list[str], fuzzy: bool = True) -> list[str]:
    """
    Normalise and deduplicate a full list of raw skills.

    Parameters
    ----------
    skills : raw list from LLM parser
    fuzzy  : enable fuzzy matching fallback

    Returns
    -------
    Deduplicated list of canonical skill names.
    """
    seen = set()
    result = []
    for s in skills:
        canonical = normalize_skill(s, fuzzy=fuzzy)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def get_canonical_vocabulary() -> list[str]:
    """Return the full list of canonical skill names (sorted)."""
    return _ALL_CANONICALS.copy()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  QUICK SELF-TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_CASES = [
        # (raw_input, expected_canonical)
        ("React.js",            "react"),
        ("ReactJS",             "react"),
        ("REACTJS",             "react"),
        ("Python 3.9",          "python"),
        ("Python3",             "python"),
        ("pytorch",             "pytorch"),
        ("PyTorch",             "pytorch"),
        ("Pytoch",              "pytorch"),   # typo
        ("Sci-kit learn",       "scikit-learn"),
        ("sklearn",             "scikit-learn"),
        ("Scikit Learn",        "scikit-learn"),
        ("TF",                  "tensorflow"),
        ("tensorflow2",         "tensorflow"),
        ("k8s",                 "kubernetes"),
        ("NodeJS",              "nodejs"),
        ("Node.js",             "nodejs"),
        ("node",                "nodejs"),
        ("PostgreSQL",          "postgresql"),
        ("Postgers",            "postgresql"),  # typo
        ("Postgres",            "postgresql"),
        ("Power BI",            "power_bi"),
        ("PowerBI",             "power_bi"),
        ("Apache Spark",        "spark"),
        ("PySpark",             "spark"),
        ("HuggingFace",         "huggingface"),
        ("Hugging Face",        "huggingface"),
        ("LangChain",           "langchain"),
        ("Vue.js",              "vue"),
        ("VueJS",               "vue"),
        ("Angular",             "angular"),
        ("AngularJS",           "angular"),
        ("MongoDB",             "mongodb"),
        ("Mongo DB",            "mongodb"),
        ("Bayesian",            "bayesian_inference"),
        ("Time Series",         "time_series"),
        ("CI/CD",               "ci_cd"),
        ("GitHub Actions",      "ci_cd"),
        ("good communication",  None),   # should be dropped
        ("teamwork",            None),   # should be dropped
        ("MS Power BI",         "power_bi"),
        ("Jupyter Notebook",    "jupyter"),
        ("GPT-4",               "gpt"),
        ("BERT",                "bert"),
        ("RoBERTa",             "bert"),
        ("YOLOv8",              "yolo"),
        ("Stable Diffusion",    "stable_diffusion"),
        ("Azure ML",            "azure_ml"),
        ("Google Vertex AI",    "vertex_ai"),
        ("Data Version Control","dvc"),
        ("Weights & Biases",    "wandb"),
        ("DBT",                 "dbt"),
    ]

    print("=" * 65)
    print(f"{'RAW INPUT':<30} {'EXPECTED':<22} {'GOT':<22} STATUS")
    print("=" * 65)

    passed = failed = 0
    for raw, expected in TEST_CASES:
        got = normalize_skill(raw)
        ok  = (got == expected)
        mark = "✅" if ok else "❌"
        print(f"{mark}  {raw:<28} {str(expected):<22} {str(got):<22}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 65)
    print(f"\nResults: {passed}/{passed+failed} passed  ({100*passed/(passed+failed):.0f}%)")
