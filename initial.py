# conda env export > environment.yml
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax import Arm, Metric, Runner, OptimizationConfig, Objective, Data
import numpy as np
import pandas as pd

# def gca(parameters):
# Conversion of Craig Schindler's Matlab code: Force_vs_Velocity_test_structures_theoretical.mat
# solves the ode for the force vs velocity of an ideal gap closer
# imports
import numpy as np
from scipy import integrate
from scipy.integrate import odeint
from scipy.integrate import RK45, solve_ivp
parameters = {
    "V": 50,
    "L": 10e-6,
    "F_load": 0,
}
# Defining Constants
eps0 = 8.85e-12
E = 170e9;
Lol = 76e-6
t_SOI = 40e-6
x0 = 4.8e-6
gf = 4.833e-6
gb = 7.75e-6
k_support = 50
N_fing = 70 # was 96, real data is 70
# gf is the nominal front gap
# gb is the nominal back gap

x_GCA = 3.833e-6
V = parameters["V"] #V
changeFactor = 100.0

A = t_SOI * Lol       # intermediate value
C = eps0 * A / gf # intermediate value
Ctot = N_fing * C
Fmin_mN = (1 / 2) * V ** 2 * Ctot / gf * 1e3

L = parameters["L"] #um
N_act = 16.0
Fmin = N_fing * (1 / 2) * V ** 2 * eps0 * t_SOI * Lol * (1 / (gf) ** 2 - 1 / (gb) ** 2)

# Load force
Fload = parameters["F_load"]*1e-6
#     Farr = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
#     Farr = np.multiply(Farr, 1e-6)
karr = np.divide(Fload, changeFactor * x_GCA)

warr_um = np.divide(karr, (E * t_SOI / (N_act * L ** 3)) ** (1 / 3) * 1e6)
warr_um_drawn = np.add(warr_um, 1)
strainarr_percent = 3 * np.multiply((np.divide(warr_um, 1e6)), changeFactor * x_GCA / (2 * N_act) / (2 * (L / 2) ** 2) * 100)

print("set values")
int_time = np.array([0, 0.5])
def pull_in(t, x):
    """
    def sim_lorenz(X, t, sigma, beta, rho):
        u, v, w = X
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return up, vp, wp
    """
    Fload_pullin = x[2]
    V = x[3]

    Fes = N_fing * (1 / 2) * (V ** 2) * eps0 * t_SOI * Lol * (1 / (gf - x[0]) ** 2 - 1 / (gb + x[0]) ** 2)
    Fd = x[1] * N_fing * 1.85e-5 * Lol * (t_SOI ** 3) /((gf - x[0]) ** 3)
    Fk = k_support * x[0]
    m = ((1350e-6 * 20e-6) + 96 * (76e-6 * 5e-6) - 56 * (8e-6 * 8e-6)) * 40e-6 * 2300
    m_spring = (600e-6 * 8e-6 * 40e-6 * 2300 * 16) * (1 / 3)  # spring effective mass, has a mass of 1/3 its actual mass
    m_spring = 0  # remove this to actually count the spring mass
    m = m + (m_spring / 3)  # shuttle mass + effective spring mass
    dxdt = [[x[1]], [(Fes - Fd - Fk - Fload_pullin) / m], [0], [0]]
    return x[1], (Fes - Fd - Fk - Fload_pullin)/m, 0, 0

t0, t1 = 0, .001  # start and end
t = np.linspace(t0, t1, 1000)  # the points of evaluation of solution\

def solve_time(ode, t, x_initial, x_dot_initial, F_load, V):
    end_time = 250e-6
    states = []
    success = False
    solve_time = -1
    solver = RK45(ode, 0, (x_initial, x_dot_initial, F_load, V), .5)
    # y = odeint(ode, (x_initial, x_dot_initial, F_load, V), t)
    while solver.t < end_time:
        solver.step()
        solver_state = solver.dense_output()
        if solver_state.y_old[0] >= 3.833e-6:
            success = True
            states.append(solver_state.y_old)
            solve_time = solver.dense_output().t
            break
        else:
            states.append(solver_state.y_old)

    return states, success, solve_time


# time initial conditions
# for pull-out x_initial = gf and xdot_initial = v_init
# for pull-in x_initial = 0 and xdot_initial = 0 (edited)
# See https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-18.pdf
# v_pullin = ( ((1/((x0-xpi)**2) - (1/(gb+xpi)**2))**-1) *2*k*(xpi)/(eps0*N*Lol*t_soi) )**(1/2) # implement eqn 4.17
t_in = solve_time(pull_in, t, x_initial=0, x_dot_initial=0, F_load=Fload, V=V) # todo integrate

v_init = (N*m_fing*v_fing + m_shut*v_shut)/m_gca
v_pullout = ( ((1/(gf**2) - (1/(gb+x0-gf)**2))**-1) *2*k*(x0-gf)/(eps0*N*Lol*t_soi) )**(1/2) # implement eqn 4.19
t_out = solve_time(pull_in, x_initial=x0-gf, x_dot_initial=v_init, F_load=Fload, V=V) # todo integrate
# confirm above gf, gf, N
# find above, x0, xpi

# return {"V_in": (v_pullin, 0.0),
#         "V_out": (v_pullout, 0.0),
#         "T_in": (t_in, 0.0),
#         "T_out": (t_out, 0.0)}








# # #### # # #### # # #### # # #### # # #### # # #### # # #### # # #### # # #### # # #### # # ####
# Load Data
# init_data = pd.read_csv('data/Valves_v0.csv')
loaded = pd.read_csv('data/gca_motor_v0.csv')

def trim_motor_data(init_data):
    empty_rows_repeat = np.where(pd.isnull(init_data))[0]
    empty_rows_repea_idxs = np.where(pd.isnull(init_data))[1]
    # empty_unique = np.unique(empty_rows_repeat)
    rows = []
    for i in range(len(init_data)):
        idxs = np.where(empty_rows_repeat==i)[0]
        if len(idxs) == 0:
            rows.append(i)

        cols_idx = empty_rows_repea_idxs[idxs]
        if 3 in cols_idx and 4 not in cols_idx:
            init_data.at[i, "PULL-IN"] = -1
            rows.append(i)

        if 3 not in cols_idx and 4 in cols_idx:
            init_data.at[i, "PULL-OUT"] = -1
            rows.append(i)

    init_data = init_data.replace("NP", value=-1)
    init_data = init_data.replace(".", value=-1)
    trimmed = init_data.iloc[rows]

    trimmed = trimmed[trimmed["PULL-OUT"] != "X"]
    trimmed = trimmed[trimmed["PULL-IN"] != "X"]
    # init_data = init_data.replace("X", value=-1)

    trimmed['PULL-IN'] = trimmed['PULL-IN'].astype(float)
    trimmed['PULL-OUT'] = trimmed['PULL-OUT'].astype(float)

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=trimmed['PULL-IN'].values, name="PULL-IN", nbinsx=50))
    fig.add_trace(go.Histogram(x=trimmed['PULL-OUT'].values, name="PULL-OUT", nbinsx=50))

    # Overlay both histograms
    fig.update_layout(barmode='overlay', plot_bgcolor='white',
                      bargap=0.2,  # gap between bars of adjacent location coordinates
                      bargroupgap=0.1  # gap between bars of the same location coordinates
                      )
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()
    return trimmed, fig

init_data, fig = trim_motor_data(loaded)
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
