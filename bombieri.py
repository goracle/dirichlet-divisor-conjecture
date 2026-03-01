import sys
sys.path.insert(0, '/home/dan/expdb/blueprint/src/python')
from literature import literature
import exponent_pair as ep
import numpy as np
from fractions import Fraction
import random

# =========================
# Setup transforms and seeds
# =========================
transforms = literature.list_hypotheses('Exponent pair transform')
transform_names = [t.name for t in transforms]
print(f"Available transforms: {transform_names}")

seed_pairs = ep.compute_exp_pairs(literature, search_depth=0)
print(f"Seed pairs: {len(seed_pairs)}")

def theta(pair):
    """Compute Dirichlet divisor exponent theta = k+l-1/2"""
    return float(pair.data.k + pair.data.l - Fraction(1,2))

# ========================================
# BFS over exponent pairs (classical baseline)
# ========================================
def bfs_exp_pairs(depth=10):
    pairs = list(seed_pairs)
    seen = {(float(h.data.k), float(h.data.l)) for h in pairs}

    for d in range(depth):
        new_pairs = []
        for pair in pairs:
            for t in transforms:
                try:
                    new_hyp = t.data.transform(pair)
                    k, l = new_hyp.data.k, new_hyp.data.l
                    if k >= 0 and l >= Fraction(1,2) and k+l <= 1:
                        key = (float(k), float(l))
                        if key not in seen:
                            seen.add(key)
                            new_pairs.append(new_hyp)
                except:
                    continue
        if not new_pairs:
            break
        pairs.extend(new_pairs)
    best = min(theta(h) for h in pairs)
    return best, pairs

# ========================================
# Evolutionary search on sequences of transforms
# ========================================
def apply_sequence(seq, seed):
    """Apply a sequence of transforms to a seed pair"""
    pair = seed
    for t in seq:
        try:
            pair = t.data.transform(pair)
        except:
            return None
    return pair

def evo_search(pop_size=50, seq_len=8, generations=1000):
    # initialize population: sequences of random transforms
    population = [[random.choice(transforms) for _ in range(seq_len)] for _ in range(pop_size)]
    seed = seed_pairs[0]
    best_theta = theta(seed)
    best_seq = None

    for g in range(generations):
        new_population = []
        for seq in population:
            # mutate: insert, delete, or swap
            new_seq = seq.copy()
            op = random.randint(0,2)
            if op == 0 and len(new_seq) < seq_len+3:  # insert
                new_seq.insert(random.randint(0,len(new_seq)), random.choice(transforms))
            elif op == 1 and len(new_seq) > 1:  # delete
                new_seq.pop(random.randint(0,len(new_seq)-1))
            elif op == 2:  # swap
                i, j = random.sample(range(len(new_seq)), 2)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

            # evaluate
            final_pair = apply_sequence(new_seq, seed)
            if final_pair is None:
                continue
            t = theta(final_pair)
            if t < best_theta:
                best_theta = t
                best_seq = new_seq
                print(f"[gen {g}] New best theta: {best_theta:.8f} via {[tr.name for tr in new_seq]}")
            new_population.append(new_seq)

        # fill population to maintain size
        while len(new_population) < pop_size:
            new_population.append([random.choice(transforms) for _ in range(seq_len)])
        population = new_population

    return best_theta, best_seq

# =========================
# Run BFS baseline
# =========================
print("\nRunning BFS depth 10...")
bfs_best, bfs_pairs = bfs_exp_pairs(depth=10)
print(f"BFS best theta: {bfs_best:.8f} ({len(bfs_pairs)} pairs)")

# =========================
# Run evolutionary search
# =========================
print("\nRunning evolutionary search...")
evo_best, evo_seq = evo_search(pop_size=50, seq_len=8, generations=500)
print(f"\nEvolutionary best theta: {evo_best:.8f}")
if evo_seq:
    print(f"Sequence: {[t.name for t in evo_seq]}")
