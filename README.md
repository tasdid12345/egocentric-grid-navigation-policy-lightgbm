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

## Reference

This project is motivated by the following paper:

> M Rosano, A Furnari, L Gulino (2021). *On embodied visual navigation in real environments through habitat*. 2020 25th International. https://ieeexplore.ieee.org/abstract/document/9413026/

Paper author: Prof. Assist. Prof. Antonino Furnari, University of Catania (furnari@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
