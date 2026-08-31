# Value of Information in Soil Measurement Acquisition for Site-Specific Fertilizer Recommendation

<p align="center">
  <a href="https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/MLxOR">
    <img src="https://img.shields.io/badge/NeurIPS%202026-MLxOR%20Workshop-6f42c1" alt="Venue">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" alt="Jupyter">
  <a href="https://colab.research.google.com/github/abdelghanibelgaid/soil-measurement-acquisition/blob/main/soil-measurement-acquisition.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

Reproducibility repository for **Value of Information in Soil Measurement Acquisition for Site-Specific Fertilizer Recommendation**, prepared for the **Second Workshop on ML×OR at NeurIPS 2026**.

## Overview

Soil measurements are commonly treated as fixed inputs to fertilizer recommendation models. Operationally, however, soil information must first be acquired through sampling and laboratory analysis, and different measurements carry different costs and uncertain downstream value.

This repository studies soil testing as an **information-allocation problem**. Rather than asking only which fertilizer action should be predicted, the analysis asks which soil measurement or measurement bundle should be acquired before the downstream fertilizer decision is made.

The framework combines machine learning with empirical value of information and budgeted subset optimization. Candidate measurements are evaluated according to their reduction in cross-validated downstream NPK reference-action loss rather than generic feature importance.

## Abstract

Soil testing creates a sequential decision problem: a recommendation system may receive management context before laboratory measurements, while each additional soil test has a monetary cost and uncertain decision value. A value-of-information formulation is developed for fertilizer decision support. For a measurement subset, cross-validated downstream action loss defines empirical risk; marginal value of information is the reduction in that loss, and a budgeted subset problem selects measurements with minimum risk subject to laboratory cost. A multi-environment benchmark uses 18 tomato environments from Latin America with soil pH, organic matter, phosphorus, potassium, and recorded extra N-P-K fertilizer amounts. The recorded fertilizer vector is treated strictly as a historical reference action, not as a causal agronomic optimum. Under leave-one-environment-out validation with a standardized multi-output ridge model, organic matter is the best singleton and reduces normalized action loss by 3.48\% relative to a no-soil baseline. Its paired 95\% bootstrap interval spans zero ($-0.1235$ to $0.2079$), indicating substantial sampling uncertainty. The complete four-measurement panel is 7.30\% worse than organic matter alone and 3.57\% worse than the no-soil baseline. Across 2,612 cost--budget cases generated from 256 relative-cost vectors, positive-value-of-information-per-cost acquisition matches the exact budget oracle in 97.05\% of cases; the remaining cases arise from joint P+K value not detectable from singleton marginal gains. The exact oracle leaves available budget unspent in 78.71\% of cases. The results show that measurement acquisition should depend on downstream value rather than affordability alone and that exact subset optimization can expose complementarities missed by myopic acquisition.

## Research question

The analysis asks:

> Which soil measurement, or measurement bundle, should be acquired when soil information is costly and downstream fertilizer-decision fidelity is the objective?

For an acquired measurement set $S$, cross-validated risk is denoted by

$$
\mathcal{R}(S).
$$

The empirical marginal value of information of an unobserved measurement $j$ is

$$
\Delta(j\mid S)=\mathcal{R}(S)-\mathcal{R}(S\cup\{j\}).
$$

Positive values indicate improved held-out reference-action fidelity; negative values indicate that the additional measurement degrades performance under the validation distribution.

Under measurement budget $B$, the exact acquisition problem is

$$
S^*(B)=\arg\min_{S:\,C(S)\le B}\mathcal{R}(S).
$$

Because only four candidate soil measurements are considered, all 16 subsets can be enumerated exactly.

## Public benchmark

The empirical benchmark uses the public `agridat::ortiz.tomato.covs` dataset derived from Ortiz, Crossa, Vargas, and Izquierdo (2007), *Studying the Effect of Environmental Variables on the Genotype × Environment Interaction of Tomato*, *Euphytica* 153:119–134.

The dataset contains **18 tomato environments in Latin America** with soil, site, management, climate, and fertilizer-management variables.

### Candidate soil measurements

- soil pH
- organic matter (OM)
- soil phosphorus (P)
- soil potassium (K)

### Free management context

- irrigation (`Irr`)
- trimming (`Trim`)
- driving (`Driv`)

Latitude and longitude are preserved in the bundled public file but excluded from the primary predictive baseline. The public table reports a longitude for environment E05 that is geographically inconsistent with its documented environment label. No inferred coordinate correction is introduced.

### Reference fertilizer actions

- extra nitrogen, `ExN`
- extra phosphorus, `ExP`
- extra potassium, `ExK`

The recorded extra N--P--K amounts are treated strictly as **historical reference actions**. They are not interpreted as experimentally verified fertilizer optima.

## Experimental design

The central model is a standardized multi-output ridge regression with

$$
\alpha=10.
$$

Each of the 18 environments is held out once using leave-one-environment-out validation. Prediction error is evaluated with normalized NPK action loss:

$$
\ell_i(S)=\frac{1}{3}\sum_{t\in\{N,P,K\}}\frac{|a_{it}-\hat a_{it}(S)|}{\operatorname{IQR}(a_t)}.
$$

The notebook evaluates all

$$
2^4=16
$$

possible soil-measurement subsets on identical held-out environments. Uncertainty in singleton value of information is estimated using **10,000 paired bootstrap resamples**.

## What the notebook reproduces

The single notebook implements the full computational workflow:

1. Loads the bundled public tomato dataset, with a public URL fallback.
2. Audits the environment coordinates and defines the coordinate-free primary context.
3. Fits standardized multi-output ridge regression under leave-one-environment-out validation.
4. Evaluates all 16 soil-measurement subsets.
5. Computes normalized downstream NPK reference-action loss.
6. Computes singleton empirical value of information.
7. Performs 10,000 paired bootstrap resamples over held-out environments.
8. Identifies the exact best subset at each measurement count.
9. Enumerates all 256 heterogeneous relative-cost vectors and 2,612 attainable cost--budget cases.
10. Compares the exact budget oracle with a positive-VOI-per-cost sequential policy.
11. Evaluates a public laboratory-menu cost illustration.
12. Runs ridge-penalty sensitivity analysis.
13. Regenerates all paper and supplementary figures and machine-readable results.
14. Executes scientific assertions matching the manuscript claims.

## Main empirical results

Under the primary coordinate-free context, the exact subset frontier is:

| Measurements | Best subset | LOEO normalized action loss | Change vs. no-soil |
|---:|---|---:|---:|
| 0 | none | 1.4429 | 0.00% |
| 1 | OM | 1.3927 | +3.48% |
| 2 | OM + K | 1.4168 | +1.81% |
| 3 | OM + P + K | 1.4461 | -0.22% |
| 4 | pH + OM + P + K | 1.4944 | -3.57% |

Organic matter is the best singleton. Adding measurements does not monotonically improve held-out decision fidelity: the complete four-measurement panel is **7.30% worse than OM alone** and **3.57% worse than the no-soil baseline**.

## Value-of-information uncertainty

For organic matter, the empirical singleton value of information is

$$
\Delta(\mathrm{OM}\mid\varnothing)=0.0502
$$

normalized-loss units, with paired 95% bootstrap interval

$$
[-0.1235,\;0.2079].
$$

The interval crosses zero. The observed singleton ranking therefore should not be interpreted as a universal agronomic hierarchy.

## Budgeted measurement acquisition

Each of the four candidate measurements receives a relative cost from

$$
\{0.5,1,2,4\},
$$

producing **256 cost vectors** and **2,612 attainable cost--budget cases**.

The sequential policy maximizes positive marginal VOI per unit cost and stops when no feasible measurement has positive marginal VOI.

- exact-oracle agreement: **97.05%**;
- mean regret: **0.0000176** normalized-loss units;
- maximum regret: **0.0005968**;
- exact oracle leaves available budget unspent in **78.71%** of cases.

The 77 mismatch cases (**2.95%**) are informative: the exact oracle selects **P+K** even though neither P nor K has positive singleton VOI from the no-soil baseline. This exposes measurement complementarity that a myopic singleton policy cannot discover.

Across all 2,612 exact budget problems, the selected oracle subsets are:

| Oracle subset | Cases | Fraction |
|---|---:|---:|
| OM | 2,023 | 77.45% |
| none | 512 | 19.60% |
| P + K | 77 | 2.95% |

## Sensitivity analysis

Ridge penalties

$$
\alpha\in\{0.1,1,10,100\}
$$

are evaluated. OM remains the highest-value singleton at all four penalties, although the magnitude of its empirical VOI changes substantially. This supports the qualitative ranking under the tested regularization grid while preserving substantial sample uncertainty.

## Data provenance

Primary dataset:

```text
agridat::ortiz.tomato.covs
```

Public documentation:

```text
https://kwstat.github.io/agridat/reference/ortiz.tomato.html
```

Public CSV mirror:

```text
https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/csv/agridat/ortiz.tomato.covs.csv
```

The repository preserves the public CSV unchanged. Additional provenance and the coordinate-quality decision are documented in `data/PROVENANCE.md`.

## Running the notebook

### Google Colab

Use the **Open in Colab** badge at the top of this README after the notebook has been committed as:

```text
soil-measurement-acquisition.ipynb
```

### Local execution

Clone the repository:

```bash
git clone https://github.com/abdelghanibelgaid/soil-measurement-acquisition.git
cd soil-measurement-acquisition
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook soil-measurement-acquisition.ipynb
```

The bundled public dataset makes the primary analysis network-independent after dependencies are installed. If the bundled CSV is absent, the notebook falls back to the public Rdatasets mirror.

## Repository structure

```text
soil-measurement-acquisition/
├── soil-measurement-acquisition.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── ortiz_tomato_covs.csv
│   └── PROVENANCE.md
├── results/
│   ├── subset_results.csv
│   ├── oracle_by_k.csv
│   ├── singleton_voi_bootstrap.csv
│   ├── greedy_equal_cost.csv
│   ├── random_order.csv
│   ├── cost_grid_results.csv
│   ├── cost_oracle_distribution.csv
│   ├── ridge_sensitivity.csv
│   ├── lab_menu_results.csv
│   └── paper_quantities.json
└── figures/
    ├── figure1_frontier.png
    ├── figure2_singleton_voi.png
    ├── figureS1_lab_cost_menu.png
```

The notebook is the computational source of truth. The committed `results/` and `figures/` directories provide immediately inspectable outputs and are overwritten when the notebook is rerun.

## Reproducibility scope

This repository reproduces a **public-data methodological benchmark for soil measurement acquisition**.

Important interpretation boundaries include:

- only 18 environments are available;
- fertilizer variables are historical management observations rather than randomized fertilizer-response optima;
- empirical value of information is defined through held-out reference-action loss;
- latitude and longitude are excluded from the primary predictive context because of a public coordinate-quality anomaly;
- the analysis does not estimate causal fertilizer effects;
- the analysis does not claim improvements in crop yield, farmer profit, nutrient-use efficiency, or environmental outcomes;
- measurement rankings remain sample- and model-dependent;
- prospective agronomic validation would require trials containing pre-treatment soil measurements, randomized nutrient rates, crop response, and measurement costs.

The benchmark should therefore be interpreted as evidence for the **information-acquisition formulation**, not as a prescriptive soil-testing hierarchy.

## Citation

The paper is prepared for workshop submission. A final BibTeX citation can be added after the review process.

For now, please cite the repository title:

> *Value of Information in Soil Measurement Acquisition for Site-Specific Fertilizer Recommendation*. Submission to the Second Workshop on ML×OR at NeurIPS 2026.

## Venue

**Workshop:** Second Workshop on ML×OR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making  
**Conference:** 40th Conference on Neural Information Processing Systems, NeurIPS 2026  
**Location:** Atlanta, Georgia, USA  
**Status:** Submission ready

Submission page: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/MLxOR

## Related resources

- `agridat` dataset documentation: https://kwstat.github.io/agridat/reference/ortiz.tomato.html
- Public Rdatasets mirror: https://vincentarelbundock.github.io/Rdatasets/
- *Machine Learning-Based Optimization of Site-Specific NPK Fertilizer Recommendation*: https://doi.org/10.1016/j.atech.2026.101823

---

**Note:** The repository accompanies a methodological study of the value of soil information for fertilizer decision support. The results motivate explicit evaluation of measurement value and acquisition cost before soil variables are assumed to be freely available to a recommendation system.