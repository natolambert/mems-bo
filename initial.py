# conda env export > environment.yml
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax import Arm, Metric, Runner, OptimizationConfig, Objective, Data
import numpy as np
import pandas as pd

# Load Data
init_data = pd.read_csv('data/Valves_v0.csv')
constraints = pd.read_csv('data/Valves_constraints.csv')

# Create Variables
# opt_list = ["N (fingers #)", "a (um)", "Resistance at 1 uA (kohm)"]
opt_list = ["N (fingers #)", "a (um)"]
metrics = ['Energy']
filter_fail = init_data['Failed'] != 1
data_success = init_data[filter_fail]
data_full = data_success[opt_list + metrics]
data_params = data_success[opt_list]


class MEMsMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = eval_params(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


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
        sub["log_scale"] = bool(constraints[o][3])
        parameters.append(sub)

        # Other parameter types
        # choice_param = ChoiceParameter(name="choice", values=["foo", "bar"], parameter_type=ParameterType.STRING)
        # fixed_param = FixedParameter(name="fixed", value=[True], parameter_type=ParameterType.BOOL)

    return parameters


parameters = gen_parameters(init_data, constraints, opt_list)

ax_client = AxClient()
ax_client.create_experiment(
    name="Thermo-mechnaical Valve",
    parameters=parameters,
    objective_name="Energy",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=None,  # Optional.
    outcome_constraints=None,  # Optional.
)
exp = ax_client._experiment

def eval_params(parameterization):
    data_eval = data_full
    for key, val in parameterization.items():
        data_eval = data_eval[data_eval[key] == val]
    results = data_eval[metrics]
    mean = np.mean(results)
    sem = np.std(results)
    return mean, sem

optimization_config = OptimizationConfig(
    objective=Objective(
        metric=MEMsMetric(name="base"),
        minimize=True,
    ),
)


class MyRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


exp.runner = MyRunner()
exp.optimization_config = optimization_config

# exp.new_batch_trial()
print(f"Loading parameter evaluations")
ran = []
for i, (idx, val) in enumerate(data_params.iterrows()):
    p = dict()
    for label in opt_list:
        p[label] = val[label]
    # exp.trials[0].add_arm(Arm(name='Batch MEMs', parameters=p))
    if p not in ran:
        exp.new_trial().add_arm(Arm(name=f"Batch MEMs {i}", parameters=p))
        ran.append(p)
        print(f" - {len(ran)} Parameters {p}")

    else:
        continue
    # ax_client.attach_trial(parameters=p)

for t in exp.trials:
    exp.trials[t].run()

# Pull out of API for more specific iteration
batch = 10
print(f"Running Batch GP+EI optimization of {batch} samples")
# Reinitialize GP+EI model at each step with updated data.
data = exp.fetch_data()
print(data.df)
gpei = Models.BOTORCH(experiment=exp, data=data)
generator = gpei.gen(batch)

exp.new_batch_trial(generator_run=generator)
new_trial = len(exp.trials) - 1
print(f"New Candidate Designs")
for arm in exp.trials[new_trial].arms:
    print(arm)

# generator_run = gpei.gen(5)
# experiment.new_batch_trial(generator_run=generator_run)

#
# quit()
# sobol = Models.SOBOL(search_space=experiment.search_space)
# generator_run = sobol.gen(5)
#
# batch = exp.new_trial(generator_run=gpei.gen(1))
#
# quit()

# for i in range(25):
#     parameters, trial_index = ax_client.get_next_trial()
#     # Local evaluation here can be replaced with deployment to external system.
#     ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
#     # _, trial_index = ax_client.get_next_trial()
#     ax_client.log_trial_failure(trial_index=trial_index)
#
# ax_client.get_trials_data_frame().sort_values('trial_index')
# best_parameters, values = ax_client.get_best_parameters()

from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
plot = plot_contour(model=gpei,
                param_x=opt_list[0],
                param_y=opt_list[1],
                metric_name="base",)
render(plot)
ax_client.generation_strategy.model = gpei
init_notebook_plotting(offline=True)
# render(ax_client.get_contour_plot())
render(ax_client.get_contour_plot(param_x=opt_list[0], param_y=opt_list[0]))#, metric_name=base))
# render(ax_client.get_optimization_trace(objective_optimum=hartmann6.fmin))  # Objective_optimum is optional.

ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.
restored_ax_client = AxClient.load_from_json_file()  # For custom filepath, pass `filepath` argument.
