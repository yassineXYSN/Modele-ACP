"""
Training Pipeline — Phase 1 → 2 → 3 + Evaluation
===================================================
Reads a CV dataset (JSONL), builds TF-IDF, trains PCA on the TRAIN split,
then evaluates profile-classification accuracy and reconstruction error on
the held-out TEST split.

Outputs
-------
models/pca_model.pkl          – trained PCA (sklearn)
models/tfidf_vectorizer.pkl   – fitted TF-IDF vectorizer
models/scaler.pkl             – fitted StandardScaler
models/skills_vectors.pkl     – (n_skills, n_components) latent coords per skill
models/skill_names.pkl        – ordered list of skill names (index → name)
models/cv_vectors.pkl         – (n_TRAIN, n_components) latent coords (train only)
models/meta.json              – variance, n_components, skill list
models/eval_report.json       – test-set accuracy, reconstruction error, silhouette

Usage
-----
  python train_pipeline.py                          # uses data/cvs_combined.jsonl
  python train_pipeline.py --data data/merged_cvs.jsonl
  python train_pipeline.py --data data/merged_cvs.jsonl --test-size 0.2 --min-freq 5
"""

import json, os, joblib
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, silhouette_score
)
from normalizer import normalize_skill_list

# ─────────────────────────────────────────────────────────────────
# CONFIG  (overridable via CLI)
# ─────────────────────────────────────────────────────────────────
DATA_PATH       = "data/cvs_combined.jsonl"
MODELS_DIR      = "models"
MIN_SKILL_FREQ  = 5       # skills appearing fewer times are dropped (noise)
TARGET_VARIANCE = 0.95    # PCA keeps enough components to explain 95% variance
TEST_SIZE       = 0.20    # 20% held out for evaluation
RANDOM_STATE    = 42


# ─────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
def _load_records(path: str) -> list[dict]:
    """Load records from JSONL or CSV (profile + skills columns)."""
    p = path.lower()
    if p.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        records = []
        for _, row in df.iterrows():
            profile = str(row.get("profile", "unknown"))
            skills_raw = str(row.get("skills", ""))
            skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
            records.append({"profile": profile, "skills_raw": skills_list})
        return records
    else:   # default: JSONL
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def phase1_load_and_clean(path: str):
    print("\n📂  Phase 1 — Loading & cleaning …")
    records = _load_records(path)

    cleaned_skills, profiles = [], []
    for r in records:
        skills = normalize_skill_list(r["skills_raw"], fuzzy=True)
        cleaned_skills.append(skills)
        profiles.append(r.get("profile", "unknown"))

    # Frequency filter — drop skills that appear too rarely
    flat  = [s for doc in cleaned_skills for s in doc]
    freq  = Counter(flat)
    filtered = [
        [s for s in doc if freq[s] >= MIN_SKILL_FREQ]
        for doc in cleaned_skills
    ]
    # Remove CVs that have no skills left after filtering
    valid = [(r, sk, p) for r, sk, p in zip(records, filtered, profiles) if len(sk) >= 2]
    records, filtered, profiles = zip(*valid) if valid else ([], [], [])
    records, filtered, profiles = list(records), list(filtered), list(profiles)

    n_before = len([k for k, v in freq.items()])
    n_after  = len([k for k, v in freq.items() if v >= MIN_SKILL_FREQ])
    print(f"   Total CVs loaded      : {len(records)}")
    print(f"   Unique skills (raw)   : {n_before}")
    print(f"   Unique skills (kept)  : {n_after}  (min_freq={MIN_SKILL_FREQ})")

    prof_dist = Counter(profiles)
    print(f"\n   Profile distribution:")
    for prof, cnt in sorted(prof_dist.items(), key=lambda x: -x[1]):
        pct = cnt / len(profiles) * 100
        print(f"     {prof:<30} {cnt:>4}  ({pct:.1f}%)")

    return records, filtered, profiles


# ─────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────
def split_data(records, filtered_skills, profiles, test_size=TEST_SIZE):
    print(f"\n✂️   Train/Test Split — {int((1-test_size)*100)}% train / {int(test_size*100)}% test")
    idx = list(range(len(records)))

    # Stratified split by profile so each profile appears in both sets
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=RANDOM_STATE,
            stratify=profiles
        )
    except ValueError:
        # Fallback: some profiles may have too few samples for stratification
        print("   ⚠️  Stratified split failed (too few samples per profile) — using random split")
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=RANDOM_STATE
        )

    print(f"   Train : {len(train_idx)}  CVs")
    print(f"   Test  : {len(test_idx)}  CVs")

    train_skills  = [filtered_skills[i] for i in train_idx]
    test_skills   = [filtered_skills[i] for i in test_idx]
    train_profiles = [profiles[i] for i in train_idx]
    test_profiles  = [profiles[i] for i in test_idx]

    return train_idx, test_idx, train_skills, test_skills, train_profiles, test_profiles


# ─────────────────────────────────────────────────────────────────
# PHASE 2 — TF-IDF  (fitted on TRAIN only)
# ─────────────────────────────────────────────────────────────────
def phase2_build_matrix(train_skills, test_skills):
    print("\n🧮  Phase 2 — Building TF-IDF matrix (fit on train) …")

    train_docs = [" ".join(sk) for sk in train_skills]
    test_docs  = [" ".join(sk) for sk in test_skills]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[^\s]+",
        sublinear_tf=True,
        min_df=MIN_SKILL_FREQ,
    )
    X_train = vectorizer.fit_transform(train_docs).toarray()
    X_test  = vectorizer.transform(test_docs).toarray()
    skill_names = vectorizer.get_feature_names_out()

    print(f"   Vocabulary size : {len(skill_names)} skills")
    print(f"   Train matrix    : {X_train.shape}")
    print(f"   Test  matrix    : {X_test.shape}")
    return X_train, X_test, vectorizer, skill_names


# ─────────────────────────────────────────────────────────────────
# PHASE 3 — PCA  (fitted on TRAIN only)
# ─────────────────────────────────────────────────────────────────
def phase3_train_pca(X_train, X_test):
    print("\n🧠  Phase 3 — Training PCA (fit on train) …")

    scaler   = StandardScaler(with_mean=True, with_std=True)
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)          # transform with train statistics

    # Auto-select n_components using TRAIN variance only
    pca_full = PCA(random_state=RANDOM_STATE).fit(Xs_train)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp   = int(np.searchsorted(cumvar, TARGET_VARIANCE)) + 1
    n_comp   = min(n_comp, Xs_train.shape[1] - 1)

    print(f"   n_components = {n_comp}  →  {cumvar[n_comp-1]*100:.1f}% variance (train)")

    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(Xs_train)

    Z_train = pca.transform(Xs_train)    # (n_train, n_comp)
    Z_test  = pca.transform(Xs_test)     # (n_test,  n_comp)

    # Skill latent coords: PCA loadings transpose
    skill_vectors = pca.components_.T   # (n_skills, n_comp)

    # Test-set reconstruction error
    X_recon   = pca.inverse_transform(Z_test)
    recon_mse = float(np.mean((Xs_test - X_recon) ** 2))
    recon_var = float(1 - recon_mse / np.var(Xs_test))

    print(f"   Test reconstruction MSE      : {recon_mse:.6f}")
    print(f"   Test reconstruction R²       : {recon_var*100:.1f}%")

    return pca, scaler, Z_train, Z_test, skill_vectors, n_comp, recon_mse, recon_var


# ─────────────────────────────────────────────────────────────────
# PHASE 4 — EVALUATION
# ─────────────────────────────────────────────────────────────────
def phase4_evaluate(Z_train, Z_test, train_profiles, test_profiles,
                    skill_vectors, n_comp):
    print("\n📊  Phase 4 — Evaluating model quality …")

    le = LabelEncoder()
    y_train = le.fit_transform(train_profiles)
    y_test  = le.transform([p if p in le.classes_ else "unknown"
                             for p in test_profiles])

    # ── KNN profile classifier ────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(Z_train, y_train)
    y_pred = knn.predict(Z_test)

    test_acc = accuracy_score(y_test, y_pred)

    # Cross-validated accuracy on train set (5-fold)
    cv_scores = cross_val_score(knn, Z_train, y_train, cv=5, scoring="accuracy")
    cv_acc    = float(cv_scores.mean())
    cv_std    = float(cv_scores.std())

    print(f"\n   KNN Profile Classifier (k=5, cosine)")
    print(f"   ┌─────────────────────────────────────────")
    print(f"   │  Test  accuracy       : {test_acc*100:.1f}%")
    print(f"   │  5-fold CV accuracy   : {cv_acc*100:.1f}% ± {cv_std*100:.1f}%")
    print(f"   └─────────────────────────────────────────")

    print(f"\n   Classification report (test set):")
    labels_present = np.unique(np.concatenate([y_test, y_pred]))
    class_names    = le.inverse_transform(labels_present)
    report_str = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=class_names,
        zero_division=0,
    )
    for line in report_str.split("\n"):
        print(f"   {line}")

    # ── Confusion matrix ─────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)

    # ── Silhouette score ─────────────────────────────────────────
    # Use first 2 components for silhouette (same as visualization)
    sil_score = None
    all_Z     = np.vstack([Z_train, Z_test])
    all_y     = np.concatenate([y_train, y_test])
    n_classes = len(np.unique(all_y))
    if n_classes >= 2 and len(all_Z) > n_classes:
        try:
            sil_score = float(silhouette_score(all_Z[:, :2], all_y, metric="euclidean"))
            print(f"\n   Silhouette score (2D)  : {sil_score:.4f}  "
                  f"({'good' if sil_score > 0.3 else 'fair' if sil_score > 0.1 else 'poor'} separation)")
        except Exception:
            pass

    eval_report = {
        "test_accuracy":    round(test_acc, 4),
        "cv_accuracy_mean": round(cv_acc, 4),
        "cv_accuracy_std":  round(cv_std, 4),
        "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
        "n_train":          len(Z_train),
        "n_test":           len(Z_test),
        "n_components":     n_comp,
        "classes":          list(le.classes_),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(class_names),
        "cv_scores_per_fold": [round(s, 4) for s in cv_scores.tolist()],
    }

    return eval_report, le, knn


# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
def save_artifacts(pca, scaler, vectorizer, skill_names,
                   Z_train, skill_vectors, records,
                   n_comp, recon_mse, recon_var, eval_report,
                   train_profiles=None):
    print(f"\n💾  Saving artefacts to {MODELS_DIR}/ …")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(pca,             f"{MODELS_DIR}/pca_model.pkl")
    joblib.dump(scaler,          f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(vectorizer,      f"{MODELS_DIR}/tfidf_vectorizer.pkl")
    joblib.dump(list(skill_names), f"{MODELS_DIR}/skill_names.pkl")
    joblib.dump(Z_train,           f"{MODELS_DIR}/cv_vectors.pkl")
    joblib.dump(skill_vectors,     f"{MODELS_DIR}/skills_vectors.pkl")

    # ── Profile centroids (for Service 4 — profile recommendation) ──
    if train_profiles is not None:
        profile_centroids = {}
        train_profiles_arr = list(train_profiles)
        for profile in set(train_profiles_arr):
            mask = [i for i, p in enumerate(train_profiles_arr) if p == profile]
            profile_centroids[profile] = Z_train[mask].mean(axis=0).tolist()
        joblib.dump(profile_centroids, f"{MODELS_DIR}/profile_centroids.pkl")
        joblib.dump(train_profiles_arr, f"{MODELS_DIR}/profile_labels.pkl")

    # Build cumvar for meta
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    meta = {
        "n_candidates":        len(records),
        "n_skills":            len(skill_names),
        "n_components":        n_comp,
        "explained_variance":  float(cumvar[n_comp - 1]),
        "reconstruction_mse":  recon_mse,
        "reconstruction_r2":   recon_var,
        "skill_list":          list(skill_names),
        "top_skills_per_component": {},
        "variance_per_component": pca.explained_variance_ratio_.tolist(),
    }
    for i in range(min(n_comp, 10)):
        loadings = pca.components_[i]
        top_idx  = np.argsort(np.abs(loadings))[::-1][:5]
        meta["top_skills_per_component"][f"PC{i+1}"] = [skill_names[j] for j in top_idx]

    with open(f"{MODELS_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(f"{MODELS_DIR}/eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    print("   ✅  pca_model.pkl")
    print("   ✅  scaler.pkl")
    print("   ✅  tfidf_vectorizer.pkl")
    print("   ✅  skill_names.pkl")
    print("   ✅  cv_vectors.pkl  (train set)")
    print("   ✅  skills_vectors.pkl")
    print("   ✅  profile_centroids.pkl  (Service 4)")
    print("   ✅  profile_labels.pkl     (Service 4)")
    print("   ✅  meta.json")
    print("   ✅  eval_report.json")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    records, filtered_skills, profiles = phase1_load_and_clean(DATA_PATH)

    (train_idx, test_idx,
     train_skills, test_skills,
     train_profiles, test_profiles) = split_data(
        records, filtered_skills, profiles, test_size=TEST_SIZE
    )

    X_train, X_test, vectorizer, skill_names = phase2_build_matrix(
        train_skills, test_skills
    )

    (pca, scaler,
     Z_train, Z_test,
     skill_vectors, n_comp,
     recon_mse, recon_var) = phase3_train_pca(X_train, X_test)

    eval_report, le, knn = phase4_evaluate(
        Z_train, Z_test, train_profiles, test_profiles,
        skill_vectors, n_comp
    )

    save_artifacts(
        pca, scaler, vectorizer, skill_names,
        Z_train, skill_vectors, records,
        n_comp, recon_mse, recon_var, eval_report,
        train_profiles=train_profiles,
    )

    print(f"""
╔══════════════════════════════════════════════════╗
║  🎉  Training complete!
║
║  Dataset     : {DATA_PATH}
║  CVs         : {len(records)}  ({len(train_idx)} train / {len(test_idx)} test)
║  Skills      : {len(skill_names)}
║  Components  : {n_comp}  ({eval_report['cv_accuracy_mean']*100:.0f}% → variance)
║
║  📈  Model Quality
║  ├─ Test accuracy    : {eval_report['test_accuracy']*100:.1f}%
║  ├─ CV accuracy      : {eval_report['cv_accuracy_mean']*100:.1f}% ± {eval_report['cv_accuracy_std']*100:.1f}%
║  ├─ Reconstruction R²: {recon_var*100:.1f}%
║  └─ Silhouette (2D)  : {eval_report.get('silhouette_score') or 'N/A'}
╚══════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Train the Skill Intelligence PCA model."
    )
    ap.add_argument("--data",      default=DATA_PATH,
                    help="Path to JSONL training file (default: data/cvs_combined.jsonl)")
    ap.add_argument("--min-freq",  type=int, default=MIN_SKILL_FREQ,
                    help="Minimum skill frequency to keep (default: 5)")
    ap.add_argument("--test-size", type=float, default=TEST_SIZE,
                    help="Fraction of data to hold out for evaluation (default: 0.2)")
    ap.add_argument("--models-dir", default=MODELS_DIR,
                    help="Output directory for model artefacts (default: models/)")
    args = ap.parse_args()

    DATA_PATH  = args.data
    MIN_SKILL_FREQ = args.min_freq
    TEST_SIZE  = args.test_size
    MODELS_DIR = args.models_dir
    main()
