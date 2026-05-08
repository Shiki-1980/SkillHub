"""
Task C: Evolutionary Adversarial Skill Generation (EASG)
=========================================================
Generates adversarial skill variants that evade Task B detectors while
preserving risk semantics. Based on:
  - AOP-Mal (Xu et al., 2025): evolutionary optimization + priority actions
  - GA Text Attack: multi-objective genetic optimization
  - ICLT: TF-IDF guided perturbation strategies

Pipeline:
  Load Task B data → Select seeds → Genetic Algorithm → Evaluation
"""

import json
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/task_c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ============================================================
# 1. DATA & MODEL LOADING
# ============================================================

def load_data_and_labels():
    """Load skills data and Task B labels."""
    df = pd.read_csv(DATA_DIR / "skills_raw.csv")
    df["text"] = (df["name"].fillna("") + " " + df["description"].fillna("") + " "
                  + df["actions"].fillna("") + " " + df["permissions"].fillna(""))

    # Load labels from Task B
    results = pd.read_csv(Path("output/task_b/anomaly_results.csv"))
    df["label"] = results["weak_label"].values
    return df


def build_or_load_models(df):
    """Rebuild Task B detection models (fast to train, avoids pickle issues)."""
    # TF-IDF
    tfidf_vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                stop_words="english", sublinear_tf=True, max_df=0.8, min_df=3)
    tfidf_matrix = tfidf_vec.fit_transform(df["text"])

    # Structural features (minimal version for Task C)
    from src.task_b_anomaly_detection import extract_structural_features, rule_based_labeling
    structural = extract_structural_features(df)
    structural_scaled = StandardScaler().fit_transform(structural.values.astype(float))

    # Combined
    X_all = hstack([tfidf_matrix, structural_scaled]).tocsr()

    # Labels
    labels = df["label"].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Isolation Forest
    iso = IsolationForest(n_estimators=300, contamination=0.1, random_state=42, n_jobs=-1)
    iso.fit(tfidf_matrix)

    # LOF
    svd = TruncatedSVD(n_components=80, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix.toarray())
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1, novelty=True, n_jobs=-1)
    lof.fit(tfidf_reduced)

    # XGBoost
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(X_all, y)

    models = {
        "tfidf_vec": tfidf_vec,
        "tfidf_matrix": tfidf_matrix,
        "structural": structural,
        "structural_scaled": structural_scaled,
        "X_all": X_all,
        "iso": iso,
        "lof": lof,
        "svd": svd,
        "xgb": xgb,
        "label_encoder": le,
    }

    return models


# ============================================================
# 2. PERTURBATION OPERATORS
# ============================================================

# Risk-related term mappings for O₅ (Risk Downgrade)
RISK_DOWNGRADE_MAP = {
    "danger": "caution",
    "warning": "notice",
    "critical": "important",
    "must not": "should avoid",
    "immediately": "promptly",
    "urgent": "timely",
    "hidden": "internal",
    "silent": "quiet",
    "bypass": "streamline",
    "disable": "adjust",
    "inject": "insert",
    "exploit": "utilize",
    "payload": "content",
    "backdoor": "alternate access",
    "exfiltrate": "transfer",
    "steal": "collect",
    "hack": "access",
    "attack": "test",
    "malicious": "custom",
    "obfuscate": "reorganize",
}

# Benign context phrases for O₄ (Context Injection)
BENIGN_CONTEXTS = [
    "This skill follows best practices for security and transparency.",
    "All operations are logged and can be reviewed by the user.",
    "Designed for legitimate development and testing purposes only.",
    "Requires user confirmation before executing any operation.",
    "Compatible with standard security audit workflows.",
    "This is a standard utility tool for developers.",
    "Follows the principle of least privilege.",
    "All network requests are made with explicit user consent.",
    "Recommended by the security community for safe usage.",
    "Open source and fully auditable.",
]


class PerturbationOperators:
    """Six perturbation operators for skill text modification."""

    def __init__(self, df_normal, tfidf_vec):
        self.df_normal = df_normal  # normal skills for context injection
        self.tfidf_vec = tfidf_vec   # for TF-IDF guided word selection
        self.normal_texts = df_normal["text"].tolist()

    def o1_synonym_substitution(self, text, strength=0.3):
        """
        Replace high-TF-IDF words with close alternatives.
        Uses simple dictionary-based replacement (reproducible).
        """
        words = text.split()
        n_replace = max(1, int(len(words) * strength))

        # Get TF-IDF scores to find important words
        try:
            tfidf_scores = self.tfidf_vec.transform([text]).toarray()[0]
            feature_names = self.tfidf_vec.get_feature_names_out()
            word_scores = {}
            for i, w in enumerate(words):
                w_clean = w.lower().strip(".,;:!?()[]{}")
                if w_clean in feature_names:
                    idx = list(feature_names).index(w_clean)
                    word_scores[i] = tfidf_scores[idx]
        except Exception:
            word_scores = {i: 1.0 for i in range(len(words))}

        # Sort by importance and replace top N
        sorted_indices = sorted(word_scores.keys(), key=lambda i: word_scores[i], reverse=True)
        to_replace = sorted_indices[:n_replace]

        for i in to_replace:
            if i < len(words):
                old_word = words[i].lower().strip(".,;:!?()[]{}")
                if old_word in RISK_DOWNGRADE_MAP:
                    words[i] = RISK_DOWNGRADE_MAP[old_word]

        return " ".join(words)

    def o2_instruction_expansion(self, text, factor=0.3):
        """
        Expand text with verbose but benign-looking instructions.
        Dilutes the density of suspicious content.
        """
        lines = text.split(". ")
        n_expand = max(1, int(len(lines) * factor))

        filler_phrases = [
            "Check if input parameters are valid",
            "Log the operation for auditing purposes",
            "Display a confirmation prompt to the user",
            "Verify that all prerequisites are met",
            "Format the output for readability",
            "Cache results for improved performance",
            "Handle edge cases gracefully",
            "Return a structured response",
        ]

        for _ in range(n_expand):
            insert_pos = random.randint(0, len(lines) - 1)
            filler = random.choice(filler_phrases)
            lines.insert(insert_pos, filler)

        return ". ".join(lines)

    def o3_permission_obfuscation(self, text, strength=0.3):
        """Restructure permission-like patterns to appear more benign."""
        perm_patterns = [
            (r"\[warn\]", "[ok]"),
            (r"\[danger\]", "[warn]"),
            (r"Shell: 需要.*?;", "Shell: 需要 standard shell access;"),
            (r"网络: 需要.*?;", "Network: 需要 local network access only;"),
        ]

        result = text
        n_applied = 0
        n_target = max(1, int(len(perm_patterns) * strength * 2))

        for pat, repl in perm_patterns:
            if n_applied >= n_target:
                break
            if random.random() < strength:
                result = re.sub(pat, repl, result, count=1)
                n_applied += 1

        return result

    def o4_context_injection(self, text, strength=0.3):
        """Inject benign context from normal skills to mask malicious intent."""
        if random.random() < strength:
            # Add a benign context phrase at the beginning
            benign = random.choice(BENIGN_CONTEXTS)

            # Or borrow from a real normal skill description
            if self.normal_texts and random.random() < 0.3:
                normal_text = random.choice(self.normal_texts)
                first_sentence = normal_text.split(".")[0]
                if len(first_sentence) < 100:
                    benign = first_sentence

            # Inject at beginning or end
            if random.random() < 0.5:
                return benign + ". " + text
            else:
                return text + ". " + benign

        return text

    def o5_risk_downgrade(self, text, strength=0.4):
        """Replace risk-related terms with softer alternatives."""
        words = text.split()
        n_replace = max(1, int(len(words) * strength))

        replaceable = []
        for i, w in enumerate(words):
            w_lower = w.lower().strip(".,;:!?()[]{}")
            if w_lower in RISK_DOWNGRADE_MAP:
                replaceable.append(i)

        n_actual = min(n_replace, len(replaceable))
        for i in random.sample(replaceable, n_actual):
            w_lower = words[i].lower().strip(".,;:!?()[]{}")
            if w_lower in RISK_DOWNGRADE_MAP:
                words[i] = RISK_DOWNGRADE_MAP[w_lower]

        return " ".join(words)

    def o6_structural_mimicry(self, text, strength=0.3):
        """
        Adjust text structure to resemble normal skill patterns.
        Normal skills have: desc ~157 chars, actions ~256 chars.
        """
        if len(text) > 400 and random.random() < strength:
            # Trim to look more like a normal skill
            trim_point = int(300 + random.gauss(100, 50))
            trim_point = max(100, min(len(text), trim_point))
            text = text[:trim_point]

        if len(text) < 100 and random.random() < strength:
            # Expand short text with benign filler
            text += ". " + random.choice(BENIGN_CONTEXTS)

        return text

    def apply(self, text, operator_id, strength=0.3):
        """Apply a specific operator to the text."""
        operators = {
            0: self.o1_synonym_substitution,
            1: self.o2_instruction_expansion,
            2: self.o3_permission_obfuscation,
            3: self.o4_context_injection,
            4: self.o5_risk_downgrade,
            5: self.o6_structural_mimicry,
        }
        if operator_id in operators:
            return operators[operator_id](text, strength)
        return text


# ============================================================
# 3. GENETIC ALGORITHM
# ============================================================

class Individual:
    """Represents an adversarial variant with a sequence of operators."""

    def __init__(self, original_text, gene_length=None):
        self.original_text = original_text
        self.gene_length = gene_length or random.randint(1, 4)
        self.genes = []  # [(operator_id, strength), ...]
        for _ in range(self.gene_length):
            self.genes.append((random.randint(0, 5), random.uniform(0.2, 0.6)))
        self.variant_text = None
        self.fitness_values = None  # (evasion_rate, semantic_sim, structural_sim)

    def express(self, operators_obj):
        """Apply gene sequence to produce variant text."""
        text = self.original_text
        for op_id, strength in self.genes:
            text = operators_obj.apply(text, op_id, strength)
        self.variant_text = text
        return text


class EASG:
    """Evolutionary Adversarial Skill Generation framework."""

    def __init__(self, models, operators, df_seeds, pop_size=100, generations=30):
        self.models = models
        self.operators = operators
        self.df_seeds = df_seeds
        self.pop_size = pop_size
        self.generations = generations

        # Pre-compute detection on all normal data for reference
        self._setup_detectors()

    def _setup_detectors(self):
        """Pre-compute reference values for fitness evaluation."""
        self.iso = self.models["iso"]
        self.lof = self.models["lof"]
        self.svd = self.models["svd"]
        self.xgb = self.models["xgb"]
        self.tfidf_vec = self.models["tfidf_vec"]
        self.structural = self.models["structural"]
        self.structural_scaled = self.models["structural_scaled"]
        self.X_all = self.models["X_all"]
        self.le = self.models["label_encoder"]

    def _detect(self, text):
        """Run all 3 detectors on a text. Returns detection results dict."""
        # TF-IDF features
        tfidf = self.tfidf_vec.transform([text])

        # Isolation Forest (on TF-IDF)
        iso_score = float(self.iso.decision_function(tfidf)[0])
        iso_pred = -1 if iso_score < 0 else 1

        # LOF (on PCA-reduced TF-IDF)
        tfidf_dense = tfidf.toarray()
        lof_reduced = self.svd.transform(tfidf_dense)
        lof_score = float(self.lof.decision_function(lof_reduced)[0])
        lof_pred = -1 if lof_score < 0 else 1

        # XGBoost: use TF-IDF + mean structural features
        struct_feats = self.structural_scaled.mean(axis=0, keepdims=True)
        X_sample = hstack([tfidf, struct_feats]).tocsr()
        xgb_pred = int(self.xgb.predict(X_sample)[0])
        xgb_label = str(self.le.inverse_transform([xgb_pred])[0])
        is_anomaly = xgb_label != "normal"

        return {
            "iso_anomaly": iso_pred == -1,
            "lof_anomaly": lof_pred == -1,
            "xgb_anomaly": is_anomaly,
            "iso_score": iso_score,
            "lof_score": lof_score,
        }

    def _evasion_rate(self, text):
        """Fraction of detectors fooled (higher = better evasion)."""
        det = self._detect(text)
        fooled = 0
        if not det["iso_anomaly"]:
            fooled += 1
        if not det["lof_anomaly"]:
            fooled += 1
        if not det["xgb_anomaly"]:
            fooled += 1
        return fooled / 3.0

    def _semantic_similarity(self, text1, text2):
        """Cosine similarity of TF-IDF vectors."""
        try:
            v1 = self.tfidf_vec.transform([text1])
            v2 = self.tfidf_vec.transform([text2])
            return float(cosine_similarity(v1, v2)[0, 0])
        except Exception:
            return 0.5

    def _structural_similarity(self, text1, text2):
        """Length-ratio based structural similarity."""
        len1, len2 = len(text1), len(text2)
        ratio = min(len1, len2) / max(len1, len2, 1)
        return ratio

    def _fitness(self, individual):
        """Multi-objective fitness: (evasion, semantic_sim, structural_sim)."""
        if individual.variant_text is None:
            individual.express(self.operators)

        f1 = self._evasion_rate(individual.variant_text)
        f2 = self._semantic_similarity(individual.original_text, individual.variant_text)
        f3 = self._structural_similarity(individual.original_text, individual.variant_text)

        return (f1, f2, f3)

    def _pareto_dominates(self, fit_a, fit_b):
        """Check if fit_a Pareto-dominates fit_b (maximization)."""
        return (all(a >= b for a, b in zip(fit_a, fit_b)) and
                any(a > b for a, b in zip(fit_a, fit_b)))

    def _tournament_select(self, population, fitnesses, k=3):
        """Tournament selection based on Pareto dominance."""
        candidates = random.sample(list(enumerate(population)), k)
        best_idx, best_ind = candidates[0]

        for idx, ind in candidates[1:]:
            if self._pareto_dominates(fitnesses[idx], fitnesses[best_idx]):
                best_idx, best_ind = idx, ind
            elif not self._pareto_dominates(fitnesses[best_idx], fitnesses[idx]):
                # Neither dominates — pick by evasion rate
                if fitnesses[idx][0] > fitnesses[best_idx][0]:
                    best_idx, best_ind = idx, ind

        return best_ind

    def _crossover(self, parent1, parent2):
        """Two-point crossover on gene sequences."""
        if len(parent1.genes) < 2 or len(parent2.genes) < 2:
            child_genes = parent1.genes[:1] + parent2.genes[:1]
        else:
            p1 = random.randint(0, min(len(parent1.genes), len(parent2.genes)) - 1)
            p2 = random.randint(p1 + 1, min(len(parent1.genes), len(parent2.genes)))
            child_genes = parent1.genes[:p1] + parent2.genes[p1:p2] + parent1.genes[p2:]

        child = Individual(parent1.original_text)
        child.genes = child_genes[:random.randint(1, 5)]  # Cap gene length
        child.gene_length = len(child.genes)
        return child

    def _mutate(self, individual, mutation_rate=0.3):
        """Random gene mutation."""
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                op_id = random.randint(0, 5)
                strength = random.uniform(0.2, 0.6)
                individual.genes[i] = (op_id, strength)

        # Possibly add/remove a gene
        if random.random() < mutation_rate * 0.5:
            if len(individual.genes) < 5 and random.random() < 0.5:
                individual.genes.append((random.randint(0, 5), random.uniform(0.2, 0.6)))
            elif len(individual.genes) > 1:
                individual.genes.pop(random.randint(0, len(individual.genes) - 1))

        individual.gene_length = len(individual.genes)
        individual.variant_text = None
        return individual

    def run(self, seed_text, seed_idx, verbose=False):
        """Run genetic algorithm for a single seed skill."""
        # Initialize population
        population = [Individual(seed_text) for _ in range(self.pop_size)]

        # Evolve
        best_front = []
        all_results = []

        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                fit = self._fitness(ind)
                ind.fitness_values = fit
                fitnesses.append(fit)

            # Track Pareto front
            for i, ind in enumerate(population):
                dominated = False
                for j, other_fit in enumerate(fitnesses):
                    if i != j and self._pareto_dominates(other_fit, fitnesses[i]):
                        dominated = True
                        break
                if not dominated:
                    best_front.append((ind, fitnesses[i]))

            # Create next generation
            new_population = []

            # Elitism: keep top 10% by evasion rate
            elite_count = max(2, self.pop_size // 10)
            sorted_by_evasion = sorted(enumerate(population), key=lambda x: fitnesses[x[0]][0], reverse=True)
            for i in range(elite_count):
                new_population.append(population[sorted_by_evasion[i][0]])

            # Fill rest via selection + crossover + mutation
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                child = self._crossover(p1, p2)
                child = self._mutate(child, mutation_rate=0.3)
                new_population.append(child)

            population = new_population

            if verbose and gen % 10 == 0:
                avg_f1 = np.mean([f[0] for f in fitnesses])
                max_f1 = max(f[0] for f in fitnesses)
                avg_sim = np.mean([f[1] for f in fitnesses])
                print(f"    Gen {gen:3d}: avg_evasion={avg_f1:.3f}  max_evasion={max_f1:.3f}  avg_sim={avg_sim:.3f}")

        # Get best result (highest evasion with sim > 0.5)
        final_fitnesses = [self._fitness(ind) for ind in population]
        valid = [(ind, fit) for ind, fit in zip(population, final_fitnesses) if fit[1] > 0.5]
        if not valid:
            valid = list(zip(population, final_fitnesses))

        valid.sort(key=lambda x: x[1][0], reverse=True)
        best_ind, best_fit = valid[0]

        return {
            "idx": seed_idx,
            "original_text": seed_text[:2000],
            "variant_text": best_ind.variant_text[:2000],
            "genes": best_ind.genes,
            "fitness": {"evasion_rate": best_fit[0], "semantic_sim": best_fit[1], "structural_sim": best_fit[2]},
            "original_detection": self._detect(seed_text),
            "variant_detection": self._detect(best_ind.variant_text),
        }


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("  Task C: Evolutionary Adversarial Skill Generation (EASG)")
    print("=" * 70)

    # Load
    print("\n[1/5] Loading data & models...")
    df = load_data_and_labels()
    print(f"  {len(df)} skills loaded")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    # Select seeds: malicious + unsafe that ARE detected
    print("\n[2/5] Building detection models...")
    models = build_or_load_models(df)
    print(f"  Models: IF + LOF + XGBoost ready")

    # Pre-screen seeds: must have at least 1 detector firing
    def is_detected(text):
        tfidf = models["tfidf_vec"].transform([text])
        iso_score = float(models["iso"].decision_function(tfidf)[0])
        tfidf_dense = tfidf.toarray()
        lof_reduced = models["svd"].transform(tfidf_dense)
        lof_score = float(models["lof"].decision_function(lof_reduced)[0])
        struct_feats = models["structural_scaled"].mean(axis=0, keepdims=True)
        X_sample = hstack([tfidf, struct_feats]).tocsr()
        xgb_pred = int(models["xgb"].predict(X_sample)[0])
        xgb_label = str(models["label_encoder"].inverse_transform([xgb_pred])[0])
        n_detected = (iso_score < 0) + (lof_score < 0) + (xgb_label != "normal")
        return n_detected, iso_score, lof_score, xgb_label

    candidates = df[df["label"].isin(["malicious", "unsafe"])].copy()
    candidates["n_detected"] = 0
    candidates["detected_by"] = ""
    for idx in candidates.index:
        n_det, iso_s, lof_s, xgb_l = is_detected(candidates.loc[idx, "text"])
        candidates.loc[idx, "n_detected"] = n_det
        detectors = []
        if iso_s < 0: detectors.append("IF")
        if lof_s < 0: detectors.append("LOF")
        if xgb_l != "normal": detectors.append("XGB")
        candidates.loc[idx, "detected_by"] = ",".join(detectors)

    # Select seeds with at least 1 detector firing
    detected_seeds = candidates[candidates["n_detected"] >= 1].head(20)
    print(f"  Candidates: {len(candidates)} total")
    print(f"  Detected (≥1): {len(detected_seeds)}")
    print(f"  Detection breakdown:")
    for n in range(4):
        cnt = (candidates["n_detected"] == n).sum()
        if cnt > 0:
            print(f"    {n}/3 detectors: {cnt}")

    seeds = detected_seeds
    if len(seeds) == 0:
        print("  WARNING: No detected seeds found, using all malicious/unsafe")
        seeds = candidates.head(20)

    # Build operators
    print("\n[3/5] Initializing perturbation operators...")
    df_normal = df[df["label"] == "normal"]
    operators = PerturbationOperators(df_normal, models["tfidf_vec"])
    print(f"  6 operators: Synonym, Expansion, Permission Obfuscation, "
          f"Context Injection, Risk Downgrade, Structural Mimicry")

    # Run GA
    print(f"\n[4/5] Running Genetic Algorithm (pop={100}, gen={30})...")
    print(f"  Note: 30 seeds × 30 gens × 100 pop = long runtime")
    print(f"  Running with pop=50, gen=20 for feasibility...")

    ga = EASG(models, operators, seeds, pop_size=50, generations=20)

    all_results = []
    for i, (idx, row) in enumerate(seeds.iterrows()):
        print(f"\n  [{i+1}/{len(seeds)}] Seed: {row['name'][:50]}... ({row['label']})")
        result = ga.run(row["text"], int(idx), verbose=True)

        orig_det = result["original_detection"]
        var_det = result["variant_detection"]
        orig_evaded = sum([not orig_det["iso_anomaly"], not orig_det["lof_anomaly"], not orig_det["xgb_anomaly"]])
        var_evaded = sum([not var_det["iso_anomaly"], not var_det["lof_anomaly"], not var_det["xgb_anomaly"]])

        print(f"    Result: evasion {orig_evaded}/3 → {var_evaded}/3, "
              f"sem_sim={result['fitness']['semantic_sim']:.3f}")

        all_results.append(result)

    # Summary statistics
    print(f"\n[5/5] Evaluation summary...")

    initial_evasions = []
    final_evasions = []
    sims = []

    for r in all_results:
        od = r["original_detection"]
        vd = r["variant_detection"]
        init_ev = sum([not od["iso_anomaly"], not od["lof_anomaly"], not od["xgb_anomaly"]])
        fin_ev = sum([not vd["iso_anomaly"], not vd["lof_anomaly"], not vd["xgb_anomaly"]])
        initial_evasions.append(init_ev)
        final_evasions.append(fin_ev)
        sims.append(r["fitness"]["semantic_sim"])

    n_seeds = len(all_results)
    n_improved = sum(1 for i, f in zip(initial_evasions, final_evasions) if f > i)
    n_fully_evaded = sum(1 for f in final_evasions if f == 3)

    print(f"\n  Seeds: {n_seeds}")
    print(f"  Improved evasion: {n_improved}/{n_seeds} ({n_improved/n_seeds*100:.1f}%)")
    print(f"  Fully evaded (3/3): {n_fully_evaded}/{n_seeds} ({n_fully_evaded/n_seeds*100:.1f}%)")
    print(f"  Avg initial evasion: {np.mean(initial_evasions):.2f}/3")
    print(f"  Avg final evasion:   {np.mean(final_evasions):.2f}/3")
    print(f"  Avg ∆ evasion:       {np.mean(final_evasions) - np.mean(initial_evasions):.2f}")
    print(f"  Avg semantic sim:    {np.mean(sims):.3f}")

    # Operator effectiveness
    op_counts = defaultdict(int)
    for r in all_results:
        for op_id, _ in r["genes"]:
            op_counts[op_id] += 1

    op_names = ["Synonym", "Expansion", "PermissionOb", "ContextInj", "RiskDown", "StructMimic"]
    print(f"\n  Operator usage:")
    for op_id in sorted(op_counts.keys()):
        print(f"    O{op_id+1} {op_names[op_id]}: {op_counts[op_id]} times")

    # Save results
    output_path = OUTPUT_DIR / "adversarial_results.json"
    # Clean for JSON serialization
    clean_results = []
    for r in all_results:
        clean_results.append({
            "idx": r["idx"],
            "original_text": r["original_text"],
            "variant_text": r["variant_text"],
            "genes": [[int(g[0]), float(g[1])] for g in r["genes"]],
            "fitness": {k: float(v) for k, v in r["fitness"].items()},
            "original_detection": {k: bool(v) if isinstance(v, (np.bool_, bool)) else float(v)
                                    for k, v in r["original_detection"].items()},
            "variant_detection": {k: bool(v) if isinstance(v, (np.bool_, bool)) else float(v)
                                   for k, v in r["variant_detection"].items()},
        })

    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "n_seeds": n_seeds,
                "n_improved": n_improved,
                "n_fully_evaded": n_fully_evaded,
                "avg_initial_evasion": float(np.mean(initial_evasions)),
                "avg_final_evasion": float(np.mean(final_evasions)),
                "avg_delta": float(np.mean(final_evasions) - np.mean(initial_evasions)),
                "avg_semantic_sim": float(np.mean(sims)),
            },
            "operator_usage": {op_names[k]: v for k, v in op_counts.items()},
            "results": clean_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Full results → {output_path}")
    print(f"\n{'='*70}")
    print("  Task C complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
