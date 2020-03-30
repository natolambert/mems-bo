# mems-bo
Bayesian Optimization of MEMs design

# Installation

To create the included environment, run this command:
```
conda env create -f environment.yml
```

# Tutorial

There is an accompanying tutorial for this code here:  https://towardsdatascience.com/design-optimization-with-ax-in-python-957b1fec776f

# Changes to add your model (simulation)

Add two files:
- add a `problem.py` file to `models/`
- add a `problem.yaml` file to `conf/model/`

Modify one file:
- add an import statement to `models/__init__.py`

# running code

From the base directory run this command:
```
python sim.py model=problem
```
Additional modifications could be needed in `conf.yaml`. Additional visualizations, model likelihood tools, and experimental data will be added soon.
