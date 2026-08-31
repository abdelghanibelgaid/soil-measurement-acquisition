# Data provenance

Primary data file: `ortiz_tomato_covs.csv`.

Source dataset: `agridat::ortiz.tomato.covs`, based on Ortiz, Crossa, Vargas, and Izquierdo (2007), *Euphytica* 153:119–134.

Public CSV mirror used for the repository:

`https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/csv/agridat/ortiz.tomato.covs.csv`

The fertilizer variables `ExN`, `ExP`, and `ExK` are recorded extra nutrient amounts and are used only as historical reference actions, not as causal agronomic optima.

## Coordinate-quality decision

The public table contains site coordinates, but the primary predictive context deliberately excludes latitude and longitude. One longitude entry (`E05`, documented as Baja Verapaz, Guatemala) is geographically inconsistent with the environment label in the public table. No inferred replacement value is introduced. The primary baseline therefore uses only the recorded management variables `Irr`, `Trim`, and `Driv`, while preserving the public CSV unchanged.
