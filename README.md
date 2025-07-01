# Echoes of AI: Investigating the Downstream Effects of AI Assistants on Software Maintainability

This repo contains a complete replication package, including raw data and scripts for the statistical analysis, for the paper "Echoes of AI: Investigating the Downstream Effects of AI Assistants on Software Maintainability" that was preregistered at the [Registered Reports Track](https://conf.researchr.org/track/icsme-2024/icsme-2024-registered-reports-track) of the 40th International Conference on Software Maintenance and Evolution (ICSME), Flagstaff, AZ, USA, Oct 6-11, 2024. After peer review at ICSME, the registered report received In-Principle Acceptance.

## Authors
- Markus Borg, CodeScene, Sweden and Lund University, Sweden
- Dave Hewett, Equal Experts, UK
- Nadim Hagatulah, Lund University, Sweden
- Noric Couderc, Lund University, Sweden
- Emma Söderberg, Lund University, Sweden
- Donal Graham, Equal Experts, South Africa
- Uttam Kini, Equal Experts, India
- Dave Farley, Continuous Delivery, UK

## Abstract
`[Context]` AI assistants, like GitHub Copilot and Cursor, are transforming software engineering. While several studies highlight productivity improvements, their impact on maintainability requires further investigation. 

`[Objective]` This study investigates whether co-development with AI assistants affects software maintainability, specifically how easily other developers can evolve the resulting source code. 

`[Method]` We conducted a two-phase controlled experiment involving 151 participants, 95% of whom were professional developers. In Phase 1, participants added a new feature to a Java web application, with or without AI assistance. In Phase 2, a randomized controlled trial, new participants evolved these solutions without AI assistance. 

`[Results]` AI-assisted development in Phase 1 led to a modest speedup in subsequent evolution and slightly higher average CodeHealth. Although neither difference was significant overall, the increase in CodeHealth was statistically significant when habitual AI users completed Phase 1. For Phase 1, we also observed a significant effect that corroborates previous productivity findings: using an AI assistant yielded a 30.7% median decrease in task completion time. Moreover, for habitual AI users, the mean speedup was 55.9%. 

`[Conclusions]` Our study adds to the growing evidence that AI assistants can effectively accelerate development. Moreover, we did not observe warning signs of degraded code-level maintainability. We recommend that future research focus on risks such as code bloat from excessive code generation and the build-up of cognitive debt as developers invest less mental effort during implementation.

## How To Cite This Work
Please cite this work as follows:

```
@article{borg_2025_echoes,
  doi = {TBD},
  author = {Borg, Markus and Hewett, Dave and Hagatulah, Nadim and Couderc, Noric and Söderberg, Emma and Graham, Donald and Kini, Uttam and Farley, Dave},  
  title = {Echoes of AI: Investigating the Downstream Effects of AI Assistants on Software Maintainability},
  publisher = {arXiv},
  year = {2025}
}
```

## Repository structure

- `data/` – contains raw and cleaned datasets
- `frequentist/` – notebooks for t-tests, chi-square, etc.
- `bayesian/` – notebooks for Bayesian modeling
- `descriptive/` – visual summaries and exploratory analysis
