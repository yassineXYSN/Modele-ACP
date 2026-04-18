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

        # Profile centroids (Service 4 — profile recommendation)
        centroids_path = models_dir / "profile_centroids.pkl"
        self.profile_centroids: dict = (
            joblib.load(centroids_path) if centroids_path.exists() else {}
        )

        # KNN index on skill vectors (used by upskilling)
        self._knn = NearestNeighbors(n_neighbors=15, metric="cosine")
        self._knn.fit(self.skill_vectors)

        # KNN index on profile centroids (used by profile_recommendation)
        if self.profile_centroids:
            self._profile_names  = list(self.profile_centroids.keys())
            self._centroid_matrix = np.array(
                [self.profile_centroids[p] for p in self._profile_names]
            )
        else:
            self._profile_names   = []
            self._centroid_matrix = np.empty((0, 0))

        # Skill name → index lookup
        self._skill_idx = {s: i for i, s in enumerate(self.skill_names)}

        print(f"✅  Model loaded — {len(self.skill_names)} skills, "
              f"{self.pca.n_components_} components, "
              f"{len(self._profile_names)} profiles")

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

    def skill_importance(
        self,
        skills: list[str] | None = None,
        top_n: int = 10,
    ) -> dict:
        """
        Return the most discriminating skills, ranked by L2 norm in latent space.

        Parameters
        ----------
        skills : list[str] | None
            If provided, rank only these specific skills (after normalisation).
            If None, return the global top_n most important skills.
        top_n : int
            How many skills to return when `skills` is None (default: 10).

        Returns
        -------
        {
          "ranked_skills": [{"skill": str, "importance_score": float}, ...],
          "top_skills_per_component": {...}
        }
        """
        norms = np.linalg.norm(self.skill_vectors, axis=1)

        if skills is not None:
            # Rank only the requested skills
            skills_norm = normalize_skill_list(skills, fuzzy=True)
            indices = [self._skill_idx[s] for s in skills_norm if s in self._skill_idx]
            # Sort by norm descending
            indices = sorted(indices, key=lambda i: norms[i], reverse=True)
        else:
            indices = list(np.argsort(norms)[::-1][:top_n])

        ranked = [
            {"skill": self.skill_names[i], "importance_score": round(float(norms[i]), 4)}
            for i in indices
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
    # SERVICE 3b — FREE SKILL EXPLORATION (no profile bias)
    # ─────────────────────────────────────────────────────────────

    def explore_skills(
        self,
        candidate_skills: list[str],
        top_n: int = 8,
        neighbors_per_skill: int = 10,
    ) -> dict:
        """
        Explore skills related to what the candidate already knows,
        WITHOUT any job-title or career-path bias.

        Difference vs upskilling()
        --------------------------
        upskilling()    → centroid of all candidate skills → single direction
                           → naturally gravitates towards a dominant profile
        explore_skills() → per-skill neighbourhood union  → all directions
                           → works even with mixed/unusual skill sets
                           → no assumption about target career

        Algorithm
        ---------
        1. For each of the candidate's skills, find its K nearest neighbours
           in the latent space independently.
        2. Pool every neighbour found across ALL base skills.
        3. Score each neighbour by:
             score = avg_similarity × (1 + 0.4 × (n_sources − 1))
           The breadth bonus rewards skills that are close to MULTIPLE
           of the candidate's skills — they are the most universally
           connected additions.
        4. Return top_n, sorted by score descending.

        The "related_to" field explains WHY each skill was recommended
        (which of the candidate's skills it is adjacent to).

        Parameters
        ----------
        candidate_skills    : list of skill strings (raw or normalised)
        top_n               : how many recommendations to return
        neighbors_per_skill : how many neighbours to fetch per base skill

        Returns
        -------
        {
          "candidate_skills": [...],
          "mode": "free_exploration",
          "recommended_skills": [
            {
              "skill": str,
              "score": float,           ← combined relevance score
              "avg_similarity": float,  ← mean cosine sim across sources
              "n_sources": int,         ← how many of your skills it's near
              "related_to": [str, ...]  ← which of your skills triggered it
            },
            ...
          ]
        }
        """
        skills_norm = normalize_skill_list(candidate_skills, fuzzy=True)
        known = [s for s in skills_norm if s in self._skill_idx]

        if not known:
            return {"error": "No recognised skills found in the vocabulary."}

        # Per-skill KNN (slightly larger than the shared self._knn)
        k = neighbors_per_skill + 1   # +1 because the skill itself may appear
        local_knn = NearestNeighbors(n_neighbors=k, metric="cosine")
        local_knn.fit(self.skill_vectors)

        # Collect: neighbour → list of (similarity, source_skill)
        neighbour_hits: dict[str, list[tuple[float, str]]] = {}

        for base in known:
            vec = self.skill_vectors[self._skill_idx[base]].reshape(1, -1)
            dists, idxs = local_knn.kneighbors(vec)

            for dist, idx in zip(dists[0], idxs[0]):
                neighbour = self.skill_names[idx]
                if neighbour in known:          # already known — skip
                    continue
                similarity = float(max(0.0, 1.0 - dist))
                if neighbour not in neighbour_hits:
                    neighbour_hits[neighbour] = []
                neighbour_hits[neighbour].append((similarity, base))

        # Score and rank
        ranked: list[dict] = []
        for skill, hits in neighbour_hits.items():
            avg_sim   = sum(s for s, _ in hits) / len(hits)
            n_sources = len(hits)
            # Breadth bonus: skills connected to many of your skills score higher
            score = avg_sim * (1.0 + 0.4 * (n_sources - 1))
            related_to = sorted(set(b for _, b in hits))
            ranked.append({
                "skill":          skill,
                "score":          round(score, 4),
                "avg_similarity": round(avg_sim, 4),
                "n_sources":      n_sources,
                "related_to":     related_to,
            })

        ranked.sort(key=lambda x: -x["score"])

        return {
            "candidate_skills":    known,
            "mode":                "free_exploration",
            "recommended_skills":  ranked[:top_n],
        }

    # ─────────────────────────────────────────────────────────────
    # SERVICE 4 — PROFILE RECOMMENDATION
    # ─────────────────────────────────────────────────────────────

    def profile_recommendation(
        self,
        candidate_skills: list[str],
        top_n: int = 3,
    ) -> dict:
        """
        Given a candidate's skills, predict which job profile(s) they best fit.

        Algorithm:
          1. Project the candidate's skills into the PCA latent space
             (via TF-IDF → StandardScaler → PCA).
          2. Compute cosine similarity between the candidate's vector and
             the centroid of each profile cluster in the training data.
          3. Return the top_n most similar profiles with confidence scores.

        Why cosine similarity to centroids?
          Each profile centroid is the "average data professional" of that type.
          The closer the candidate is to a centroid, the more their skill mix
          resembles that profile — regardless of scale (number of skills).

        Returns
        -------
        {
          "candidate_skills": [...],       ← recognised canonical skills
          "recommended_profiles": [
            {
              "rank": 1,
              "profile": "data_scientist",
              "label": "Data Scientist",
              "confidence": 0.87,          ← cosine sim, clamped [0, 1]
              "description": "..."
            },
            ...
          ]
        }
        """
        _PROFILE_DESCRIPTIONS = {
            "data_scientist":  "Analyse, modélisation et Machine Learning sur des données structurées.",
            "data_engineer":   "Construction et maintenance de pipelines de données (ETL, Spark, Airflow).",
            "ml_engineer":     "Développement et déploiement de modèles ML en production.",
            "data_analyst":    "Visualisation, reporting et exploration de données (SQL, BI tools).",
            "mlops_engineer":  "CI/CD pour le ML : Docker, Kubernetes, monitoring de modèles.",
            "nlp_engineer":    "Traitement du langage naturel : BERT, transformers, spaCy.",
            "cv_engineer":     "Vision par ordinateur : YOLO, OpenCV, traitement d'images.",
            "ai_researcher":   "Recherche fondamentale en IA : mathématiques, publications, nouveaux modèles.",
        }

        if not self.profile_centroids:
            return {"error": "Profile centroids not available. Re-run train_pipeline.py."}

        # Encode candidate
        skills_norm = normalize_skill_list(candidate_skills, fuzzy=True)
        recognised  = [s for s in skills_norm if s in self._skill_idx]

        candidate_vec = self._encode_candidate(candidate_skills).reshape(1, -1)

        # Cosine similarity vs every profile centroid
        sims = cosine_similarity(candidate_vec, self._centroid_matrix)[0]

        # Rank
        ranked_idx = np.argsort(sims)[::-1][:top_n]

        recommended = []
        for rank, idx in enumerate(ranked_idx, 1):
            profile = self._profile_names[idx]
            score   = float(sims[idx])
            recommended.append({
                "rank":        rank,
                "profile":     profile,
                "label":       profile.replace("_", " ").title(),
                "confidence":  round(max(0.0, score), 4),
                "description": _PROFILE_DESCRIPTIONS.get(profile, ""),
            })

        return {
            "candidate_skills":      recognised,
            "recommended_profiles":  recommended,
        }

    # ─────────────────────────────────────────────────────────────
    # MASTER ENDPOINT
    # ─────────────────────────────────────────────────────────────

    def get_insights(self, candidate_skills: list[str]) -> dict:
        """
        One-shot analysis combining all four services.

        Returns
        -------
        dict with keys:
          "profile_recommendation" – best matching job profiles (Service 4)
          "upskilling"             – skills to learn next           (Service 3)
          "skill_importance"       – most discriminating skills     (Service 1)
          "profile_vector"         – raw latent vector (for MongoDB storage)
        """
        return {
            "profile_recommendation": self.profile_recommendation(candidate_skills),
            "upskilling":             self.upskilling(candidate_skills),
            "skill_importance":       self.skill_importance(),
            "profile_vector":         self._encode_candidate(candidate_skills).tolist(),
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
