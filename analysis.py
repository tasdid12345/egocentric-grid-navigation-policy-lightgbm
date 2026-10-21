"""Imitation policies on random grid mazes.
Baseline: MLP on the raw egocentric window. Improved: class-balanced LightGBM on the
same window plus engineered goal-direction, distance and turn features.
Both are scored by closed-loop goal-reaching rollouts in unseen test mazes."""
import json
import os
import sys
import time
from collections import deque

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 42
SEEDS = [42, 43, 44]
N_MAZES, SIZE, VIEW, STARTS, CAP = 1500, 15, 9, 5, 100
R = VIEW // 2                                          # half window size
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])   # up, down, left, right
T0 = time.time()
rng = np.random.default_rng(SEED)


def bfs(free, goal):
    """BFS distance from goal to every cell (-1 for walls / unreachable)."""
    dist = -np.ones(free.shape, int)
    dist[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if free[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def make_maze():
    """Random grid: ~25% interior walls, solid border. Returns walls, goal, distances, starts."""
    while True:
        wall = rng.random((SIZE, SIZE)) < 0.25
        wall[[0, -1], :] = True
        wall[:, [0, -1]] = True
        cells = np.argwhere(~wall)
        goal = tuple(cells[rng.integers(len(cells))])
        dist = bfs(~wall, goal)
        cand = np.argwhere(dist >= 5)          # starts at least 5 steps from goal
        if len(cand) >= 40:
            starts = cand[rng.choice(len(cand), STARTS, replace=False)]
            return wall, goal, dist, starts


print("Generating mazes...")
mazes = [make_maze() for _ in range(N_MAZES)]
# Padded maps: 0 free, 1 wall (also outside the grid), 2 goal
P = np.ones((N_MAZES, SIZE + 2 * R, SIZE + 2 * R), np.int8)
G = np.zeros((N_MAZES, 2), int)
for i, (wall, goal, _, _) in enumerate(mazes):
    P[i, R:R + SIZE, R:R + SIZE] = wall
    P[i, goal[0] + R, goal[1] + R] = 2
    G[i] = goal

# Split ONCE by maze ID (80/20), stratified by reachable-area quartile
reach = np.array([(mz[2] >= 0).sum() for mz in mazes])
strata = np.digitize(reach, np.quantile(reach, [.25, .5, .75]))
tr_ids, te_ids = train_test_split(np.arange(N_MAZES), test_size=0.2,
                                  random_state=SEED, stratify=strata)


def windows(m, r, c):
    """9x9 egocentric windows centred on (r, c) for maze indices m."""
    off = np.arange(VIEW)
    rows = r[:, None] + off        # padded coords: cell (r, c) sits at (r+R, c+R)
    cols = c[:, None] + off
    return P[m[:, None, None], rows[:, :, None], cols[:, None, :]]


def expert_steps(ids):
    """(maze, row, col, action) for BFS-shortest-path expert steps from every start."""
    out = []
    for i in ids:
        dist, starts = mazes[i][2], mazes[i][3]
        for r, c in starts:
            while dist[r, c] > 0:
                for a, (dr, dc) in enumerate(MOVES):
                    if dist[r + dr, c + dc] == dist[r, c] - 1:
                        break
                out.append((i, r, c, a))
                r, c = r + dr, c + dc
    return np.array(out)


def raw_feat(W, g, r, c):
    """Baseline input: flattened raw egocentric window scaled to [0, 1]."""
    return W.reshape(len(W), -1) / 2.0


def eng_feat(W, g, r, c):
    """Improved input: raw window plus goal direction, distance, free-neighbour and turn cues."""
    n = len(W)
    free = W != 1
    # free flags for up, down, left, right neighbours of the agent
    nb = np.stack([free[:, R - 1, R], free[:, R + 1, R],
                   free[:, R, R - 1], free[:, R, R + 1]], 1).astype(float)
    dr, dc = g[:, 0] - r, g[:, 1] - c
    # turn cues: is the one-step move toward the goal (vertically / horizontally) free?
    v_free = np.where(dr < 0, nb[:, 0], nb[:, 1])
    h_free = np.where(dc < 0, nb[:, 2], nb[:, 3])
    extra = np.column_stack([dr, dc, np.abs(dr), np.abs(dc), np.abs(dr) + np.abs(dc),
                             np.sign(dr), np.sign(dc), nb, nb.sum(1), v_free, h_free])
    return np.hstack([W.reshape(n, -1), extra])


def steps_X(S, feat):
    """Feature matrix for expert-step array S."""
    m, r, c = S[:, 0], S[:, 1], S[:, 2]
    return feat(windows(m, r, c), G[m], r, c)


def rollout(predict, feat, ids):
    """Closed-loop rollouts from every start of the given mazes; returns goal-reaching rate."""
    ep = np.array([(i, r, c) for i in ids for r, c in mazes[i][3]])
    m, r, c = ep.T.copy()
    done = np.zeros(len(ep), bool)
    for _ in range(CAP):                       # all episodes stepped in parallel
        act = np.where(~done)[0]
        if act.size == 0:
            break
        a = predict(feat(windows(m[act], r[act], c[act]), G[m[act]], r[act], c[act]))
        nr, nc = r[act] + MOVES[a, 0], c[act] + MOVES[a, 1]
        ok = P[m[act], nr + R, nc + R] != 1    # walls block the move
        r[act] = np.where(ok, nr, r[act])
        c[act] = np.where(ok, nc, c[act])
        done[act] = (r[act] == G[m[act], 0]) & (c[act] == G[m[act], 1])
    return done.mean()


Str, Ste = expert_steps(tr_ids), expert_steps(te_ids)
print(f"Expert steps: train {len(Str)}, test {len(Ste)} ({time.time() - T0:.0f}s)")


def mlp(seed, **kw):
    return MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                         max_iter=100, random_state=seed, **kw)


def lgbm(seed, **kw):
    return lgb.LGBMClassifier(class_weight="balanced", n_estimators=300, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                              verbose=-1, **kw)


def tune(est, grid, feat):
    """Small grid search with 3-fold CV grouped by maze, on training mazes only."""
    gs = GridSearchCV(est, grid, cv=GroupKFold(n_splits=3), scoring="f1_macro")
    gs.fit(steps_X(Str, feat), Str[:, 3], groups=Str[:, 0])
    print(f"  best CV params {gs.best_params_} (CV macro-F1 {gs.best_score_:.3f})")
    return gs.best_params_


def run(name, make, params, feat):
    """Train with each seed, then score on held-out expert steps and test-maze rollouts."""
    Xtr, ytr = steps_X(Str, feat), Str[:, 3]
    Xte, yte = steps_X(Ste, feat), Ste[:, 3]
    out = {"success": [], "f1": [], "acc": []}
    for s in SEEDS:
        model = make(s, **params).fit(Xtr, ytr)
        pred = model.predict(Xte)
        out["f1"].append(f1_score(yte, pred, average="macro"))
        out["acc"].append(accuracy_score(yte, pred))
        out["success"].append(rollout(model.predict, feat, te_ids))
        print(f"  {name} seed {s}: success {out['success'][-1]:.3f} "
              f"acc {out['acc'][-1]:.3f} macroF1 {out['f1'][-1]:.3f}")
    return {k: np.array(v) for k, v in out.items()}


print("Tuning baseline MLP (alpha) on training mazes...")
bp = tune(mlp(SEED), {"alpha": [1e-4, 1e-2]}, raw_feat)
print("Tuning improved LightGBM on training mazes...")
ip = tune(lgbm(SEED), {"num_leaves": [31, 63], "learning_rate": [0.05, 0.1]}, eng_feat)

print("Training and evaluating baseline (MLP, raw window)...")
base = run("baseline", mlp, bp, raw_feat)
print("Training and evaluating improved (balanced LightGBM, engineered features)...")
imp = run("improved", lgbm, ip, eng_feat)

b, i = base["success"], imp["success"]
res = {
    "question": ("Can a class-balanced LightGBM policy on compact egocentric features beat an "
                 "MLP policy that consumes raw egocentric map images at choosing navigation "
                 "actions and reaching goals in unseen grid mazes?"),
    "method": ("Random grid mazes generated with numpy; BFS expert actions; 9x9 egocentric "
               "windows; maze-level 80/20 split; closed-loop goal-reaching rollouts (cap "
               f"{CAP}) on test mazes; seeds {SEEDS}."),
    "dataset": {
        "name": "Random grid mazes generated in code (numpy, no download)",
        "source_url": "https://numpy.org",
        "n_rows": int(len(Str) + len(Ste)),
        "n_train": int(len(Str)),
        "n_test": int(len(Ste)),
        "same_as_paper": False,
        "baseline_is_papers_method": False,
    },
    "metric_name": "goal-reaching success rate of closed-loop rollouts in unseen test mazes",
    "higher_is_better": True,
    "baseline": "scikit-learn MLPClassifier imitation policy on the raw flattened 9x9 egocentric window",
    "improved": ("Class-balanced LightGBM on the same window plus goal direction, distance, "
                 "free-neighbour and turn-cue features; small grouped-CV grid"),
    "metrics": {
        "baseline_score": float(b.mean()),
        "improved_score": float(i.mean()),
        "improvement": float(i.mean() - b.mean()),
        "baseline_score_std": float(b.std()),
        "improved_score_std": float(i.std()),
        "baseline_f1": float(base["f1"].mean()),
        "improved_f1": float(imp["f1"].mean()),
        "baseline_acc": float(base["acc"].mean()),
        "improved_acc": float(imp["acc"].mean()),
        "n_test_episodes": int(len(te_ids) * STARTS),
    },
    "notes": ("Synthetic mazes stand in for visual observations; the paper's RL-trained visual "
              "policy is not reproduced. Baseline is imitation-trained, not RL-trained. Tuning uses "
              "3-fold GroupKFold by maze on training mazes only. Only 3 seeds and one maze split, so "
              "std is across seeds, not across splits. Test starts are unseen mazes but drawn from "
              "the same generator. Improved model is class-balanced; baseline MLP is not."),
}
if not all(np.isfinite(list(res["metrics"].values()))):
    print("Non-finite metrics; failing.")
    sys.exit(1)

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Baseline MLP\n(raw window)", "Improved LightGBM\n(balanced + features)"],
       [b.mean(), i.mean()], yerr=[b.std(), i.std()], capsize=6, color=["#999999", "#3a7"])
ax.set_ylabel("Goal-reaching success (test mazes)")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("figures/success_rate.png", dpi=120)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(f"Baseline success {b.mean():.3f} +/- {b.std():.3f}; "
      f"improved {i.mean():.3f} +/- {i.std():.3f}. Wrote results.json ({time.time() - T0:.0f}s)")
