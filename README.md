# Balanced LightGBM vs. raw-window MLP for goal-reaching in grid mazes

Motivation: "On embodied visual navigation in real environments through habitat" shows that deep visual navigation policies can be trained to reach goals in realistic 3D scenes, but such pipelines are expensive and hard to inspect. This experiment asks whether a much cheaper, interpretable tabular learner on compact egocentric features can match or beat a neural policy that reads raw egocentric maps, in a controlled grid-maze setting.

Question: Can a class-balanced LightGBM policy on compact egocentric features beat an MLP policy that consumes raw egocentric map images at choosing navigation actions and reaching goals in unseen grid mazes?

Dataset: random grid mazes generated in code with numpy (no download). Expert actions are BFS shortest-path steps to a goal. Each observation is an egocentric window of wall, free and goal cells. This is not the paper's dataset. Mazes are split by maze ID into training and held-out test mazes.

Baseline: scikit-learn MLPClassifier imitating the expert on the raw flattened egocentric window. It is a stand-in for the paper's deep visual policy, not a reproduction of it.

Improved model: class-balanced LightGBM on the same window plus engineered goal-direction, distance, free-neighbour and turn-cue features, with a small grouped cross-validation grid on training mazes.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to results.json (and figures/success_rate.png).

## Results (latest run)

On synthetic random grid mazes (1,500 unseen-maze test episodes), a standard MLP imitation policy on raw egocentric map images reached 9% goal-reaching success, while a class-balanced LightGBM on compact egocentric features reached 79%, averaged over three seeds.

| Metric | Value |
|---|---|
| baseline_score | 0.09378 |
| improved_score | 0.7936 |
| improvement | 0.6998 |
| baseline_score_std | 0.00607 |
| improved_score_std | 0.00274 |
| baseline_f1 | 0.6325 |
| improved_f1 | 0.9261 |
| baseline_acc | 0.6327 |
| improved_acc | 0.9262 |
| n_test_episodes | 1500 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic mazes stand in for visual observations; the paper's RL-trained visual policy is not reproduced. Baseline is imitation-trained, not RL-trained. Tuning uses 3-fold GroupKFold by maze on training mazes only. Only 3 seeds and one maze split, so std is across seeds, not across splits. Test starts are unseen mazes but drawn from the same generator. Improved model is class-balanced; baseline MLP is not.

## Reference

This project is motivated by the following paper:

> M Rosano, A Furnari, L Gulino (2021). *On embodied visual navigation in real environments through habitat*. 2020 25th International. https://ieeexplore.ieee.org/abstract/document/9413026/

Paper author: Prof. Assist. Prof. Antonino Furnari, University of Catania (furnari@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
