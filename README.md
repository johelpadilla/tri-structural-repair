# Systemic Tau as a homeostatic trigger for structural self-repair in event-driven networks

Johel Padilla-Villanueva, DrPH  
Department of Environmental Health, Graduate School of Public Health, Medical Sciences Campus, University of Puerto Rico  
ORCID [0000-0002-5797-6931](https://orcid.org/0000-0002-5797-6931)  
Correspondence: joel.padilla2@upr.edu

The English article is archived at [https://doi.org/10.5281/zenodo.23001549](https://doi.org/10.5281/zenodo.23001549).  
The reference implementation and the trial-level records are archived at [https://doi.org/10.5281/zenodo.23001551](https://doi.org/10.5281/zenodo.23001551).

The code in `tri_reference/` is released under the MIT license. The article and the Spanish summary are distributed under the Creative Commons Attribution 4.0 International license, as stated on the first page of each PDF.

## Contents

| Path | Content |
|---|---|
| `Padilla-Villanueva_2026_Systemic_Tau_structural_self-repair.pdf` | English article |
| `Padilla-Villanueva_2026_Tau_sistemico_reparacion_estructural_resumen.pdf` | Spanish summary |
| `tri_reference/` | Reference implementation, configuration, tests and the reported run |
| `figures/` | Figure 1 in English and Spanish |

The reported comparisons use seeds 1 to 30 once. Those rows are in `tri_reference/results_confirmatory/`. A fresh run should write to another directory. The consolidation penalty used for the reported recurrent condition is 10, recorded in `tri_reference/results_confirmatory/ewc_tuning.json`. The configuration file retains 100, the value written before that selection.

Installation and the command for a new run are described in `tri_reference/README.md`.
