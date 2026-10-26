from random import choices

import numpy as np
import pyfiglet
from colorama import  Fore
import argparse
import pandas as pd
from .misc import*
#RESOURCE FILES##


from addicty import Dict

def print_ascii_art_logo():
    ascii_art = """
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .............................................................................
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .*@:. .. .. .. ..  .  .  .. .. 
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .:@@@@@@ .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. .@@@@  .+@@  .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. *@@@@@=. .. .@@=. .. .. ..  .  .  .. .. 
              ...................................@@@@@@%........@@@........................
               .. ..  .  .  .. .. .. .. .. .. ..@@@%@@...  .  . *@@ .. .. .. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  ..@@%-@@... .. .. .+@@. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. @@@.@@.. ..  .  . @%. .. .. .. .. .. .. .. ..
               .. ..  .  .  .. .. .. .. .. ..#@@ =@@.. ..  .  .:@@  .. .. .. .. .. .. .. ..
              .. .. .. .. .. :%*@@@@@@@.  .  @@*.@@-. .. .. .. @@-.. .. .. ..  .  .  .. .. 
               .. ..  .  .  .#@@. .. @@@-. ..@@. @@:.. ..  .  @@%.  .. .. .. .. .. .. .. ..
               .. ..  .  .  .. ..-@@@@@@@@@@:@@+@@@@@@-..  .%@@# .  .. .. .. .. .. .. .. ..
              .. .. .. .. ..@@@@@@@@-  .  :@@@@@@@....@@@@@@@@... .. .. .. ..  .  .  .. .. 
               .. ..  .  .  @@..@@@@@@@ .. .*@@: .. .. .@@@@%@@@@@# .. .. .. .. .. .. .. ..
              .................-@@@@:@@:.....@@..........@@@@%.=@@@@@*.....................
              .. .. .. .. .. ..@@@@@@@@.  . %@@ .. .. .. @@@.. %@@..%@@=.. ..  .  .  .. .. 
               .. ..  .  .  .. @@*=@@.. ...@@@.. .. .. ..@@:  .@@.  .@@@@... .. .. .. .. ..
              .. .. .. .. .. ..#@@ ..  .%@@@@*  .. .. ..@@@ ..@@+ ..-@@.@@#..  .  .  .. .. 
              .. .. .. .. .. .. #@@@@@@@@#.:@%  .. .. %@@=. .@@#. ..@@- .@@#.  .  .  .. .. 
               .. ..  .  .  .. .. .-=:. .. .-@@# ..*@@@@.  .@@=  ..@@+ ..@@@ .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  . *%@@@@@@@ .@@@. ..-@@:.. %@@@@  .  .  .. .. 
              .. .. .. .. .. .. .. ..  .  .  .  .. .. @@@@@ .. .@@@. ..@@@.%@. .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. *@@#.  -@@#  .#@@@ .@@:. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. @@@@@@@. ..@@%....@@. .  .  .. .. 
              .............................................%@@@@*#@@@#..=%@@-..............
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .%@@@@@@@@@@@@ .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. ..@@.  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. @@ .. .. .. .. ..
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. *. .. .. .. .. ..
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
               .. ..  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. ..
              .............................................................................
              .. .. .. .. .. .. .. ..  .  .  .  .. .. .. .. .. .. .. .. .. ..  .  .  .. .. 
"""
    print(ascii_art)


def show_ascii_art():
    # Generate ASCII Art for SwarmetriX
    ascii_art = pyfiglet.figlet_format("SearchLibrium", '5lineoblique')

    print(Fore.MAGENTA +ascii_art)
    print_ascii_art_logo()
    Fore.RESET
    #rt = ()

def introduce_package():
    print_package_guide()


def print_package_guide():
    guide = """
SearchLibrium — Discrete Choice Model Search
============================================

INSTALL
-------
    pip install SearchLibrium

QUICK START
-----------
    python -m SearchLibrium --info             # print this guide
    python -m SearchLibrium --preview_datasets # preview bundled datasets
    python -m SearchLibrium --test_search      # run a standard MNL/MXL search
    python -m SearchLibrium --test_search_nest # run a nested-logit search


HOW THE SEARCH WORKS
--------------------
SearchLibrium uses Simulated Annealing (SA) to automatically search over:
  - Which variables to include in the utility function
  - Whether parameters should be random (taste heterogeneity)
  - Which Box-Cox transformations to apply
  - Whether to use MNL, Mixed Logit, RRM, or Mixed-RRM

The search only accepts CONVERGED solutions and enforces that ALL retained
variables are statistically significant (backward elimination at each step).

Each perturbation is guaranteed to produce a genuinely different specification:
a change that only swaps a distributional family (e.g. normal → lognormal) on
an existing random variable, without changing which variables are included, is
not considered a distinct model and will trigger another perturbation attempt.


DATA FORMAT
-----------
Your dataframe must be in LONG format — one row per alternative per observation:

    obs_id | alt   | choice | var1 | var2 | ...
    -------|-------|--------|------|------|----
    1      | car   | 1      | 35   | 12   | ...
    1      | train | 0      | 60   | 8    | ...
    1      | bus   | 0      | 55   | 5    | ...
    2      | car   | 0      | 40   | 14   | ...
    ...


EXAMPLE 1 — Standard Search (MNL + Mixed Logit)
------------------------------------------------
```python
import numpy as np
import pandas as pd
from SearchLibrium import Parameters, call_siman

df = pd.read_csv("data/Swissmetro_final.csv")
varnames   = ["TIME", "COST", "HEADWAY", "SEATS"]
choice_set = np.unique(df["alt"]).tolist()

parameters = Parameters(
    criterions   = [("bic", -1)],        # minimise BIC (-1 = minimise)
    df           = df,
    varnames     = varnames,
    asvarnames   = varnames,             # all vars can be alternative-specific
    isvarnames   = [],                   # no individual-specific vars here
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["multinomial", "mixed_logit"],
    allow_random = True,                 # allow random parameters
    allow_bcvars = True,                 # allow Box-Cox transformations
    p_val        = 0.05,                 # significance threshold
    all_sig      = True,                 # enforce all variables significant
)

best = call_siman(parameters, init_sol=None, id_num=1)
print(best)
```


EXAMPLE 2 — Random Regret Minimisation (RRM) Search
----------------------------------------------------
RRM models assume travellers minimise regret rather than maximise utility.
The search can automatically explore MNL, Mixed Logit, RRM, and Mixed-RRM.

```python
import numpy as np
import pandas as pd
from SearchLibrium import Parameters, call_siman

df = pd.read_csv("data/TravelMode.csv")
varnames   = ["vcost", "travel", "wait"]
choice_set = np.unique(df["alt"]).tolist()

parameters = Parameters(
    criterions   = [("bic", -1)],
    df           = df,
    varnames     = varnames,
    asvarnames   = varnames,
    isvarnames   = [],
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["id"].values,
    ind_id       = df["id"].values,
    base_alt     = None,
    models       = ["multinomial", "random_regret", "mixed_random_regret"],
    allow_random = True,
    p_val        = 0.05,
    all_sig      = True,
)

best = call_siman(parameters, init_sol=None, id_num=1)
```


EXAMPLE 3 — Mixed-RRM with Random Parameters
---------------------------------------------
Mixed-RRM combines taste heterogeneity (random parameters) with regret
minimisation. Use `mixed_random_regret` in the models list and enable
`allow_random`:

```python
parameters = Parameters(
    criterions   = [("bic", -1)],
    df           = df,
    varnames     = ["TIME", "COST", "HEADWAY"],
    asvarnames   = ["TIME", "COST", "HEADWAY"],
    isvarnames   = [],
    choice_set   = choice_set,
    choices      = df["CHOICE"].values,
    alt_var      = df["alt"].values,
    choice_id    = df["custom_id"].values,
    ind_id       = df["ID"].values,
    base_alt     = "SM",
    models       = ["mixed_logit", "mixed_random_regret"],
    allow_random = True,                  # REQUIRED for mixed-RRM
    n_draws      = 500,                   # halton draws for simulation
    p_val        = 0.05,
    all_sig      = True,
)
best = call_siman(parameters, init_sol=None, id_num=1)
```


AVAILABLE MODELS
----------------
  "multinomial"          — Multinomial Logit (MNL), JAX-accelerated MLE
  "mixed_logit"          — Mixed Logit (simulation-based), JAX-accelerated
  "random_regret"        — Random Regret Minimisation (RRM), JAX-accelerated
  "mixed_random_regret"  — Mixed-RRM with random parameters, JAX-accelerated
  "nested_logit"         — Nested Logit (requires nests= and lambdas= kwargs)
  "ordered_logit"        — Ordered Logit

RANDOM PARAMETER DISTRIBUTIONS
-------------------------------
  "n"  — normal
  "ln" — lognormal
  "t"  — triangular
  "tn" — truncated normal
  "u"  — uniform

SEARCH PARAMETERS
-----------------
  p_val        — significance threshold for backward elimination (default 0.05)
  all_sig      — enforce all variables significant (default True)
  allow_random — allow random parameters in the search (default False)
  allow_bcvars — allow Box-Cox transformations (default False)
  allow_corvars— allow correlated random parameters (default False)
  n_draws      — number of Halton draws for mixed models (default 1000)
  maxiter      — maximum MLE iterations (default 2000)

INTERPRETING RESULTS
--------------------
  - Lower BIC / AIC = better fit-complexity tradeoff.
  - All retained variables are statistically significant (p < p_val).
  - Random parameters indicate heterogeneity in that attribute's taste.
  - RRM models are preferred when regret-avoidance drives choice behaviour.
  - Compare across model types: utility-maximising vs regret-minimising.

Policy analysis
---------------
Use estimated utilities to compare the effect of policy levers such as fares,
travel time, subsidies, regulations, or service quality changes.

Forecasting
-----------
Generate future choice shares by updating covariates with projected inputs and
recomputing probabilities under the fitted model.

Simulation
----------
Draw from fitted coefficients or class assignments to simulate behavior under
new environments, interventions, or demand conditions.

Counterfactual analysis
-----------------------
Replace observed covariates with hypothetical values and compare predicted
choices, elasticities, welfare, or segment membership.

Sensitivity analysis
--------------------
Vary one assumption at a time, such as priors, scaling, availability, or
specification, and measure how much conclusions change.

Scenario analysis
-----------------
Bundle multiple assumptions into coherent cases such as baseline, optimistic,
constrained, or disruptive futures and compare outcomes across cases.

Uncertainty analysis
--------------------
Propagate parameter uncertainty, input uncertainty, and model uncertainty into
predicted probabilities, market shares, and decision metrics.

Risk analysis
-------------
Use tail outcomes, downside scenarios, and uncertainty intervals to understand
where a policy or forecast may fail.

Decision analysis
-----------------
Translate model outputs into actions by comparing expected utility, expected
cost, regret, robustness, or value of information.

Optimization
------------
Embed model predictions inside an optimizer to choose prices, service levels,
resource allocations, or network designs subject to constraints.

Machine learning
----------------
Use SearchLibrium as a structured behavioral layer within a broader ML pipeline
for feature engineering, prediction, or segmentation.

Deep learning
-------------
Combine learned representations from neural networks with utility-based choice
layers when nonlinear feature extraction is needed.

Reinforcement learning
----------------------
Use discrete choice outputs as behavior models, reward components, or policy
constraints in sequential decision settings.

Causal inference
----------------
Use domain knowledge and counterfactual structure to test intervention effects,
while pairing SearchLibrium with careful identification assumptions.

Explainable AI
--------------
SearchLibrium supports transparent utility decomposition, making it easier to
explain why predicted choices change when inputs change.

Interpretability
----------------
Interpret coefficients through direction, magnitude, heterogeneity, class
membership, substitution patterns, and elasticity summaries.

Fairness
--------
Check whether predictions, welfare, or policy impacts differ systematically
across demographic or protected groups.

Ethics
------
Document assumptions, data provenance, modeling limits, and stakeholder harms
before using model outputs in sensitive settings.

Social impact
-------------
Assess who benefits, who is burdened, and how access, affordability, and
equity change under competing interventions.

Sustainability
--------------
Use demand shifts and scenario outputs to study resource efficiency, emissions,
and long-run system resilience.

Climate change
--------------
Analyze mode shifts, energy demand, exposure, resilience, and adaptation
strategies under climate policy or infrastructure scenarios.

Energy
------
Model technology adoption, tariff response, appliance choice, and consumer
switching under alternative pricing or reliability conditions.

Transportation
--------------
Estimate route, mode, departure-time, and service adoption behavior for network
planning, pricing, accessibility, and reliability analysis.

Urban planning
--------------
Use predicted choices to study zoning, accessibility, land use, service
coverage, and mobility impacts across neighborhoods.

Public policy
-------------
Evaluate interventions with transparent behavioral assumptions and compare
efficiency, equity, and robustness across policy options.

Economics
---------
Apply the package to demand estimation, welfare analysis, heterogeneity,
substitution, and market response questions.

Marketing
---------
Use choice models for product design, pricing, promotion, assortment, and
customer segmentation.

Psychology
----------
Study preferences, decision heuristics, context effects, and heterogeneity in
individual choice behavior.

Sociology
---------
Model how institutions, norms, inequality, and group structure shape observed
choice patterns.

Anthropology
------------
Adapt utility specifications to culturally grounded preferences, local context,
and population-specific behavioral assumptions.

Political science
-----------------
Analyze voting, participation, program uptake, coalition preferences, or policy
support through structured discrete choice frameworks.

History
-------
Use archived or reconstructed choice settings to compare how incentives and
institutions changed behavior over time.

Philosophy
----------
Use the package to formalize assumptions about preference, rationality, welfare,
ethics, and decision making, then test their implications empirically.

Helpful commands
----------------
--info              Show this full guide
--intro             Show this full guide
--preview_datasets  Preview the example datasets
--test_search       Run the standard search example
--test_search_nest  Run the nested search example

Notes
-----
- Calling info prints this full guide.
- If no command is supplied, the guide is shown automatically.
- Search results are strongest when combined with domain knowledge and external validation.
"""
    print(Fore.RESET + guide)





