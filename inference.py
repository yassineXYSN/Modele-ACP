"""
Inference Module — import this in your web app.

Usage
-----
from inference import SkillAnalyzer

analyzer = SkillAnalyzer()          # loads models once at startup

result = analyzer.get_insights(
    candidate_skills=["python", "pandas", "scikit-learn", "sql"]
)
print(result)
"""

import joblib
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from normalizer import normalize_skill, normalize_skill_list  # shared normalizer

MODELS_DIR = Path(__file__).parent / "models"


class SkillAnalyzer:
    """
    Three-in-one skill analysis engine.

    Services:
      1. skill_importance  – which skills matter most for a given role/component
      2. skill_liaison     – cosine similarity between two skills in latent space
      3. upskilling        – skills the candidate should learn next (centroid KNN)
    """

    def __init__(self, models_dir: str | Path = MODELS_DIR):
        models_dir = Path(models_dir)
        print("🔄  Loading model artefacts …")

        self.pca        = joblib.load(models_dir / "pca_model.pkl")
        self.scaler     = joblib.load(models_dir / "scaler.pkl")
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        self.skill_names: list[str]     = joblib.load(models_dir / "skill_names.pkl")
        self.skill_vectors: np.ndarray  = joblib.load(models_dir / "skills_vectors.pkl")
        self.cv_vectors: np.ndarray     = joblib.load(models_dir / "cv_vectors.pkl")

        with open(models_dir / "meta.json") as f:
            self.meta = json.load(f)

        # KNN index on skill vectors (used by upskilling)
        self._knn = NearestNeighbors(n_neighbors=15, metric="cosine")
        self._knn.fit(self.skill_vectors)

        # Skill name → index lookup
        self._skill_idx = {s: i for i, s in enumerate(self.skill_names)}

        print(f"✅  Model loaded — {len(self.skill_names)} skills, "
              f"{self.pca.n_components_} components")

    # ─────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────────

    def _encode_candidate(self, skills: list[str]) -> np.ndarray:
        """
        Turn a list of skill strings into a latent vector via TF-IDF → PCA.
        Used for profile_vector (MongoDB storage) and profile-level operations.
        """
        skills_norm = normalize_skill_list(skills, fuzzy=True)
        doc       = " ".join(skills_norm)
        tfidf_vec = self.vectorizer.transform([doc]).toarray()
        scaled    = self.scaler.transform(tfidf_vec)
        latent    = self.pca.transform(scaled)
        return latent[0]

    def _candidate_centroid(self, skills: list[str]) -> np.ndarray | None:
        """
        Compute the centroid of the known skill vectors for the given skills.

        This is the preferred approach for upskilling because it is stable
        even when the candidate has only 2–3 skills:
          - Each skill already has a well-defined position in latent space.
          - Their average (centroid) reliably represents the candidate's domain.
          - TF-IDF projection of a sparse vector can land in noisy regions.

        Returns None if no skills are found in the model vocabulary.
        """
        skills_norm = normalize_skill_list(skills, fuzzy=True)
        vectors = [
            self.skill_vectors[self._skill_idx[s]]
            for s in skills_norm
            if s in self._skill_idx
        ]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)   # (n_components,)

    # ─────────────────────────────────────────────────────────────
    # SERVICE 1 — SKILL IMPORTANCE
    # ─────────────────────────────────────────────────────────────

    def skill_importance(self, top_n: int = 10) -> dict:
        """
        Return the most discriminating skills overall.
        Ranked by L2 norm in latent space: high norm = this skill separates
        different job profiles the most.
        """
        norms   = np.linalg.norm(self.skill_vectors, axis=1)
        top_idx = np.argsort(norms)[::-1][:top_n]

        ranked = [
            {"skill": self.skill_names[i], "importance_score": round(float(norms[i]), 4)}
            for i in top_idx
        ]
        return {
            "ranked_skills": ranked,
            "top_skills_per_component": self.meta.get("top_skills_per_component", {})
        }

    # ─────────────────────────────────────────────────────────────
    # SERVICE 2 — SKILL LIAISON
    # ─────────────────────────────────────────────────────────────

    def skill_liaison(self, skill_a: str, skill_b: str) -> dict:
        """
        Measure how related two skills are in latent space.

        The raw cosine similarity can be negative when two skills belong to
        competing specialisations (e.g. Tableau vs PyTorch — one is BI,
        the other is deep learning). Negative values are clamped to 0 for
        display, but the raw value is also returned for advanced use.

        Returns
        -------
        {
          "skill_a": str,
          "skill_b": str,
          "cosine_similarity": float,   ← clamped to [0, 1]
          "raw_similarity": float,      ← actual value, can be negative
          "interpretation": str
        }
        """
        a = normalize_skill(skill_a, fuzzy=True) or skill_a.lower().strip()
        b = normalize_skill(skill_b, fuzzy=True) or skill_b.lower().strip()

        if a not in self._skill_idx:
            return {"error": f"'{skill_a}' not found in model vocabulary."}
        if b not in self._skill_idx:
            return {"error": f"'{skill_b}' not found in model vocabulary."}

        va  = self.skill_vectors[self._skill_idx[a]].reshape(1, -1)
        vb  = self.skill_vectors[self._skill_idx[b]].reshape(1, -1)
        raw = float(cosine_similarity(va, vb)[0][0])
        sim = max(0.0, raw)   # clamp — negative similarity has no HR meaning

        if raw < -0.1:
            interp = "Competitive — profiles specialise in one OR the other."
        elif raw < 0.1:
            interp = "Independent — rarely appear together (different specialisations)."
        elif sim > 0.8:
            interp = "Very strongly related — almost always co-occur."
        elif sim > 0.6:
            interp = "Strongly related — frequently seen together."
        elif sim > 0.4:
            interp = "Moderately related — often complementary."
        elif sim > 0.2:
            interp = "Weakly related — occasional overlap."
        else:
            interp = "Loosely related — some shared context."

        return {
            "skill_a":           a,
            "skill_b":           b,
            "cosine_similarity": round(sim, 4),    # clamped [0,1]
            "raw_similarity":    round(raw, 4),    # true value
            "interpretation":    interp,
        }

    # ─────────────────────────────────────────────────────────────
    # SERVICE 3 — UPSKILLING (centroid KNN)
    # ─────────────────────────────────────────────────────────────

    def upskilling(self, candidate_skills: list[str], top_n: int = 5) -> dict:
        """
        Find the skills closest to the candidate's current profile.

        Algorithm (centroid approach):
          1. Look up the latent vector of each known skill.
          2. Average them → the candidate's "centroid" in skill space.
          3. Run KNN from that centroid to find nearby skills.

        Why centroid instead of TF-IDF projection?
          TF-IDF projection of a sparse input (2–3 skills) can land in
          a noisy region of the latent space, causing irrelevant suggestions
          (e.g. recommending 'pillow' to a BI analyst with tableau/excel).
          The centroid stays inside the actual cluster of the candidate's skills.

        Falls back to TF-IDF projection if no skills are in the vocabulary.
        """
        skills_norm = normalize_skill_list(candidate_skills, fuzzy=True)
        known       = set(skills_norm)

        centroid = self._candidate_centroid(candidate_skills)
        if centroid is not None:
            query = centroid.reshape(1, -1)
        else:
            # Fallback: TF-IDF projection
            query = self._encode_candidate(candidate_skills).reshape(1, -1)

        distances, indices = self._knn.kneighbors(query)

        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            skill     = self.skill_names[idx]
            proximity = round(float(max(0.0, 1 - dist)), 4)
            if skill not in known:
                recommendations.append({"skill": skill, "proximity_score": proximity})
            if len(recommendations) >= top_n:
                break

        return {
            "candidate_skills":    list(known & set(self.skill_names)),
            "recommended_skills":  recommendations,
        }

    # ─────────────────────────────────────────────────────────────
    # MASTER ENDPOINT
    # ─────────────────────────────────────────────────────────────

    def get_insights(self, candidate_skills: list[str]) -> dict:
        """
        One-shot analysis combining all three services.

        Returns
        -------
        dict with keys: "upskilling", "skill_importance", "profile_vector"
        """
        return {
            "upskilling":       self.upskilling(candidate_skills),
            "skill_importance": self.skill_importance(),
            "profile_vector":   self._encode_candidate(candidate_skills).tolist(),
        }


# ─────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = SkillAnalyzer()

    TESTS = [
        ("BI Analyst",        ["tableau", "power_bi", "excel"]),
        ("MLOps Engineer",    ["docker", "kubernetes", "airflow"]),
        ("AI Researcher",     ["linear_algebra", "calculus"]),
        ("Data Scientist",    ["python", "pandas", "scikit-learn", "sql", "statistics"]),
    ]

    for label, skills in TESTS:
        print(f"\n{'='*60}\n  {label}: {skills}\n{'='*60}")
        result = analyzer.get_insights(skills)

        print("  Upskilling →")
        for r in result["upskilling"]["recommended_skills"]:
            print(f"    {r['skill']:<25}  {r['proximity_score']}")

    print(f"\n{'='*60}\n  Skill Liaison\n{'='*60}")
    pairs = [
        ("docker",    "kubernetes"),
        ("tableau",   "power_bi"),
        ("pytorch",   "tensorflow"),
        ("tableau",   "pytorch"),
    ]
    for a, b in pairs:
        r = analyzer.skill_liaison(a, b)
        if "error" not in r:
            print(f"  {a:<15} ↔ {b:<15}  "
                  f"sim={r['cosine_similarity']}  raw={r['raw_similarity']}  |  {r['interpretation']}")
