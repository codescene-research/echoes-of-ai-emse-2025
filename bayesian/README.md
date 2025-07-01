# Bayesian Analysis

This is a Bayesian analysis for the paper "Echoes of AI: Investigating the Downstream Effects of AI Assistants on Software Maintainability" that was preregistered at the Registered Reports Track of the 40th International Conference on Software Maintenance and Evolution (ICSME), Flagstaff, AZ, USA, Oct 6-11, 2024. 

# Setup 

In a Julia interpreter: 

```
using Pkg
Pkg.activate('.')
Pkg.instantiate()
```

Run tests with

```
Pkg.test()
```

This may take a while (we run some tests with MCMC sampling...) 

# Jupyter Notebooks

The main thing to think about when opening notebooks is to set the number of threads to 4, otherwise MCMC sampling runs slower.

Run this:
```
$ julia --project=.
julia> ENV["JULIA_NUM_THREADS"] = 4
julia> using IJulia
julia> IJulia.notebook()
```

Then a browser will open, and you can pick a notebook.
