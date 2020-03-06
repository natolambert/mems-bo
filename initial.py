import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

# conda env export > environment.yml

import pandas as pd

init_data = pd.read_csv('data/Valves_v0.csv')
constraints = pd.read_csv('data/Valves_constraints.csv')
from ax import optimize
# best_parameters, best_values, experiment, model = optimize(
#         parameters=[
#           {
#             "name": "x1",
#             "type": "range",
#             "bounds": [-10.0, 10.0],
#           },
#           {
#             "name": "x2",
#             "type": "range",
#             "bounds": [-10.0, 10.0],
#           },
#         ],
#         # Booth function
#         evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
#         minimize=True,
#     )

from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient

opt_list = ["N (fingers #)", "a (um)", ]
data_sub = init_data[1]

# TODO
# sub data, add elements to experiment
# figure out how the data is matched to a objective in a reasonable way
# sample new values / save
# visualization 
def gen_parameters(data, constraints, opt_list):
    parameters = []
    for o in opt_list:
        sub = dict()
        typ = constraints[o][2]
        sub["name"] = o
        sub["type"] = "range"
        if typ == "float":
            sub["bounds"] = [float(constraints[o][0]), float(constraints[o][1])]
        elif typ == "int":
            sub["bounds"] = [int(constraints[o][0]), int(constraints[o][1])]
        else:
            raise ValueError("Type not supported")
        sub["value_type"] = typ
        sub["log_scale"] = False
        parameters.append(sub)
    return parameters


parameters = gen_parameters(init_data, constraints, opt_list)

ax_client = AxClient()
ax_client.create_experiment(
    name="Thermo-mechnaical Valve",
    parameters=parameters,
    objective_name="hartmann6",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=None,  # Optional.
    outcome_constraints=None,  # Optional.
)

# Pull out of API for more specific iteration
exp = ax_client._experiment
batch = 10
print(f"Running Batch GP+EI optimization of {batch} samples")
# Reinitialize GP+EI model at each step with updated data.
gpei = Models.BOTORCH(experiment=exp, data=init_data)
generator = gpei.gen(5)
# generator_run = gpei.gen(5)
# experiment.new_batch_trial(generator_run=generator_run)


quit()
sobol = Models.SOBOL(search_space=experiment.search_space)
generator_run = sobol.gen(5)

batch = exp.new_trial(generator_run=gpei.gen(1))

quit()

ax_client = AxClient()
ax_client.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
    ],
    objective_name="hartmann6",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
)

# ax_client.attach_trial(parameters={"x1": 0.9, "x2": 0.9, "x3": 0.9, "x4": 0.9, "x5": 0.9, "x6": 0.9})


# import numpy as np
# def evaluate(parameters):
#     x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
#     # In our case, standard error is 0, since we are computing a synthetic function.
#     return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}
#

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    # _, trial_index = ax_client.get_next_trial()
    ax_client.log_trial_failure(trial_index=trial_index)

ax_client.get_trials_data_frame().sort_values('trial_index')
best_parameters, values = ax_client.get_best_parameters()

from ax.utils.notebook.plotting import render, init_notebook_plotting

render(ax_client.get_contour_plot())
render(ax_client.get_contour_plot(param_x="x3", param_y="x4", metric_name="l2norm"))
render(ax_client.get_optimization_trace(objective_optimum=hartmann6.fmin))  # Objective_optimum is optional.

ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.
restored_ax_client = AxClient.load_from_json_file()  # For custom filepath, pass `filepath` argument.
