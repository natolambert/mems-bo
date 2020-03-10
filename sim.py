from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax import  * #ParameterType, FixedParameter, Arm, Metric, Runner, OptimizationConfig, Objective, Data
import numpy as np
import pandas as pd

from models import jumper

import os
import sys
import hydra
import logging

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())
log = logging.getLogger(__name__)
# Create Variables
opt_list = ["N", "L"]
metrics = ['U_stored']


class MEMsMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            mean, sem = jumper(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))

def gen_search_space(cfg):
    l = []
    for key, item in cfg.space.items():
        print( key)

    return

@hydra.main(config_path='conf/conf.yaml')
def mems_exp(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    search_space = gen_search_space(cfg.problem)
    outcome_con = gen_outcome_constraints(cfg.problem)

    exp = SimpleExperiment(
            name="PID Control Robot",
            search_space=SearchSpace(search_space),
            evaluation_function=bo_rollout_wrapper,
            objective_name="Reward",
            minimize=False,
            outcome_constraints=outcome_con,
        )


    ax_client = AxClient()
    ax_client.create_experiment(
        name="Jumping Robot",
        parameters=[{
            "name": "N",
            "type": "range",
            "bounds": [1, 6],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
            {
                "name": "L",
                "type": "range",
                "bounds": [500e-6, 1.9e-2],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, defaults to False.
            },
            {
                "name": "w",
                "type": "range",
                "bounds": [4.0e-5, 4.0e-3],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, defaults to False.
            },
            # FixedParameter(name="maxF", value=15.0e-3, parameter_type=ParameterType.FLOAT),
            # FixedParameter(name="maxS", value=0.5e-2, parameter_type=ParameterType.FLOAT),
            # FixedParameter(name="maxX", value=5.0e-3, parameter_type=ParameterType.FLOAT),
            # FixedParameter(name="T", value=550.0e-6, parameter_type=ParameterType.FLOAT),
            # FixedParameter(name="E", value=170.0e9, parameter_type=ParameterType.FLOAT),
        ],
        objective_name="Energy",
        minimize=False,  # Optional, defaults to False.
        parameter_constraints=[],  # Optional.
        outcome_constraints=[],  # ["Energy >= 0"],  # Optional.
    )
    # # [
    #         OutcomeConstraint(
    #             metric=L2NormMetric(
    #                 name="Energy", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
    #             ),
    #             op=ComparisonOp.QEQ,
    #             bound=0,
    #             relative=False,
    #         )
    # #     ],
    exp = ax_client._experiment

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=MEMsMetric(name="base"),
            minimize=False,
        ),
    )



    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config

    # exp.new_batch_trial()

    for t in exp.trials:
        exp.trials[t].run()

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(exp.search_space)
    num_search = 15
    for i in range(num_search):
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()

    data = exp.fetch_data()

    num_opt = 15
    for i in range(num_opt):
        print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
        # Reinitialize GP+EI model at each step with updated data.
        gpei = Models.BOTORCH(experiment=exp, data=data)
        batch = exp.new_trial(generator_run=gpei.gen(1))
        exp.trials[len(exp.trials) - 1].run()
        data = exp.fetch_data()

    # from ax.plot.trace import optimization_trace_single_method
    # from ax.utils.notebook.plotting import render, init_notebook_plotting
    # from ax.plot.contour import plot_contour
    #
    # print(f"Running Sobol initialization trials...")
    # sobol = Models.SOBOL(exp.search_space)
    # num_search = 10
    # for i in range(num_search):
    #     exp.new_trial(generator_run=sobol.gen(1))
    #     exp.trials[len(exp.trials) - 1].run()
    #
    # # data = exp.fetch_data()
    # gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    # plot = plot_contour(model=gpei,
    #                     param_x="roll-p",
    #                     param_y="pitch-p",
    #                     metric_name="base", )
    # data = plot[0]['data']
    # lay = plot[0]['layout']
    #
    # render(plot)
    #
    # num_opt = 50
    # for i in range(num_opt):
    #     print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
    #     # Reinitialize GP+EI model at each step with updated data.
    #     batch = exp.new_trial(generator_run=gpei.gen(1))
    #     gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    #
    #     if (i % 5) == 0:
    #         plot = plot_contour(model=gpei,
    #                             param_x="roll-p",
    #                             param_y="pitch-p",
    #                             metric_name="base", )
    #         data = plot[0]['data']
    #         lay = plot[0]['layout']
    #
    #         render(plot)

    best_parameters, values = ax_client.get_best_parameters()

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
                        param_x="N",
                        param_y="L",
                        metric_name="base", )
    data = plot[0]['data']
    lay = plot[0]['layout']

    import plotly.graph_objects as go
    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image("test.pdf")

    render(plot)


if __name__ == '__main__':
    sys.exit(mems_exp())
