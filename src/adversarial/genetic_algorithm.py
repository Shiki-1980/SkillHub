"""
EASG: Evolutionary Adversarial Skill Generation.
Based on AOP-Mal (Xu 2025) + GA Text Attack (2020).
"""
import random, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


class Individual:
    def __init__(self, original_text, gene_length=None):
        self.original_text = original_text
        self.gene_length = gene_length or random.randint(1, 4)
        self.genes = [(random.randint(0, 5), random.uniform(0.2, 0.6))
                      for _ in range(self.gene_length)]
        self.variant_text = None
        self.fitness_values = None

    def express(self, operators_obj):
        text = self.original_text
        for op_id, strength in self.genes:
            text = operators_obj.apply(text, op_id, strength)
        self.variant_text = text
        return text


class EASG:
    def __init__(self, models, operators, pop_size=50, generations=20):
        self.models = models
        self.operators = operators
        self.pop_size = pop_size
        self.generations = generations
        self._setup_detectors()

    def _setup_detectors(self):
        self.iso = self.models["iso"]
        self.lof = self.models["lof"]
        self.svd = self.models["svd"]
        self.xgb = self.models["xgb"]
        self.tfidf_vec = self.models["tfidf_vec"]
        self.structural_scaled = self.models["structural_scaled"]
        self.le = self.models["label_encoder"]

    def _detect(self, text):
        tfidf = self.tfidf_vec.transform([text])
        iso_score = float(self.iso.decision_function(tfidf)[0])
        iso_pred = -1 if iso_score < 0 else 1
        lof_reduced = self.svd.transform(tfidf.toarray())
        lof_score = float(self.lof.decision_function(lof_reduced)[0])
        lof_pred = -1 if lof_score < 0 else 1
        sf = self.structural_scaled.mean(axis=0, keepdims=True)
        X = hstack([tfidf, sf]).tocsr()
        xgb_pred = int(self.xgb.predict(X)[0])
        xgb_label = str(self.le.inverse_transform([xgb_pred])[0])
        return {
            "iso_anomaly": iso_pred == -1,
            "lof_anomaly": lof_pred == -1,
            "xgb_anomaly": xgb_label != "normal",
            "iso_score": iso_score, "lof_score": lof_score,
        }

    def _evasion_rate(self, text):
        d = self._detect(text)
        return sum([not d["iso_anomaly"], not d["lof_anomaly"], not d["xgb_anomaly"]]) / 3.0

    def _semantic_similarity(self, t1, t2):
        try:
            v1, v2 = self.tfidf_vec.transform([t1]), self.tfidf_vec.transform([t2])
            return float(cosine_similarity(v1, v2)[0, 0])
        except Exception:
            return 0.5

    def _fitness(self, ind):
        if ind.variant_text is None: ind.express(self.operators)
        f1 = self._evasion_rate(ind.variant_text)
        f2 = self._semantic_similarity(ind.original_text, ind.variant_text)
        f3 = min(len(ind.original_text), len(ind.variant_text)) / max(len(ind.original_text), len(ind.variant_text), 1)
        return (f1, f2, f3)

    def _pareto_dominates(self, a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    def _tournament_select(self, pop, fits, k=3):
        idxs = random.sample(range(len(pop)), k)
        best = idxs[0]
        for i in idxs[1:]:
            if self._pareto_dominates(fits[i], fits[best]):
                best = i
            elif not self._pareto_dominates(fits[best], fits[i]) and fits[i][0] > fits[best][0]:
                best = i
        return pop[best]

    def _crossover(self, p1, p2):
        if len(p1.genes) < 2 or len(p2.genes) < 2:
            child_genes = p1.genes[:1] + p2.genes[:1]
        else:
            a = random.randint(0, min(len(p1.genes), len(p2.genes)) - 1)
            b = random.randint(a + 1, min(len(p1.genes), len(p2.genes)))
            child_genes = p1.genes[:a] + p2.genes[a:b] + p1.genes[b:]
        c = Individual(p1.original_text)
        c.genes = child_genes[:random.randint(1, 5)]
        return c

    def _mutate(self, ind, rate=0.3):
        for i in range(len(ind.genes)):
            if random.random() < rate:
                ind.genes[i] = (random.randint(0, 5), random.uniform(0.2, 0.6))
        if random.random() < rate * 0.5:
            if len(ind.genes) < 5 and random.random() < 0.5:
                ind.genes.append((random.randint(0, 5), random.uniform(0.2, 0.6)))
            elif len(ind.genes) > 1:
                ind.genes.pop(random.randint(0, len(ind.genes)-1))
        ind.variant_text = None
        return ind

    def run(self, seed_text, seed_idx, verbose=False):
        pop = [Individual(seed_text) for _ in range(self.pop_size)]
        for gen in range(self.generations):
            fits = [self._fitness(ind) for ind in pop]
            for i, ind in enumerate(pop): ind.fitness_values = fits[i]
            new_pop = []
            elite = sorted(range(len(pop)), key=lambda i: fits[i][0], reverse=True)
            for i in range(max(2, self.pop_size // 10)):
                new_pop.append(pop[elite[i]])
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fits)
                p2 = self._tournament_select(pop, fits)
                new_pop.append(self._mutate(self._crossover(p1, p2)))
            pop = new_pop
            if verbose and gen % 10 == 0:
                print(f"    Gen {gen:3d}: avg_evasion={np.mean([f[0] for f in fits]):.3f}  "
                      f"max_evasion={max(f[0] for f in fits):.3f}")
        final_fits = [self._fitness(ind) for ind in pop]
        valid = [(i, f) for i, f in zip(pop, final_fits) if f[1] > 0.5] or list(zip(pop, final_fits))
        valid.sort(key=lambda x: x[1][0], reverse=True)
        best, best_fit = valid[0]
        return {
            "idx": seed_idx, "original_text": seed_text[:2000],
            "variant_text": best.variant_text[:2000], "genes": best.genes,
            "fitness": {"evasion_rate": best_fit[0], "semantic_sim": best_fit[1], "structural_sim": best_fit[2]},
            "original_detection": self._detect(seed_text),
            "variant_detection": self._detect(best.variant_text),
        }
