from ax import (
    ComparisonOp,
    ParameterType, Parameter, RangeParameter, ChoiceParameter,
    FixedParameter, OutcomeConstraint, SimpleExperiment, Models,
    Arm, Metric, Runner, OptimizationConfig, Objective, Data,
    SearchSpace
)

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
        if item.value_type == 'float':
            typ = ParameterType.FLOAT
        elif item.value_type == 'int':
            typ = ParameterType.INT
        elif item.value_type == 'bool':
            typ == Parameter.BOOL
        else:
            raise ValueError("invalid search space value type")

        if item.type == 'range':
            ss = RangeParameter(
                name=key, parameter_type=typ, lower=item.bounds[0], upper=item.bounds[1], log_scale=item.log_scale,
            )
        elif item.type == 'fixed':
            ss = FixedParameter(name=key, value=item.bounds, parameter_type=typ)
        elif item.type == 'choice':
            ss = ChoiceParameter(name=key, parameter_type=typ, values=item.bounds)
        else:
            raise ValueError("invalid search space parameter type")
        l.append(ss)
    return l


def gen_outcome_constraints(cfg):
    l = []
    for key, item in cfg.constraints.items():
        if item.type == 'GEQ':
            op = ComparisonOp.GEQ
        elif item.type == 'LEQ':
            op = ComparisonOp.LEQ
        else:
            raise ValueError("Improper Constraint Type")
        cons = OutcomeConstraint(
            metric=MEMsMetric(name=key),
            op=op,
            bound=item.value,
            relative=False,
        )
        l.append(cons)
    return l


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
        evaluation_function=jumper,
        objective_name="Energy",
        minimize=False,
        outcome_constraints=outcome_con,
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=MEMsMetric(name="Energy"),
            minimize=False,
        ),
    )

    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config
    from ax.plot.trace import optimization_trace_single_method
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    from ax.plot.contour import plot_contour

    print(f"Running {cfg.bo.random} Sobol initialization trials...")
    sobol = Models.SOBOL(exp.search_space)
    num_search = cfg.bo.random
    for i in range(num_search):
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()

    # data = exp.fetch_data()
    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    plot = plot_contour(model=gpei,
                        param_x="N",
                        param_y="L",
                        metric_name="Energy", )
    data = plot[0]['data']
    lay = plot[0]['layout']

    render(plot)

    num_opt = cfg.bo.optimized
    for i in range(num_opt):
        print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
        # Reinitialize GP+EI model at each step with updated data.
        batch = exp.new_trial(generator_run=gpei.gen(1))
        gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

        if (i % 5) == 0:
            plot = plot_contour(model=gpei,
                                param_x="N",
                                param_y="L",
                                metric_name="Energy", )
            data = plot[0]['data']
            lay = plot[0]['layout']

            render(plot)

    objective_means = np.array([[exp.trials[trial].objective_mean] for trial in exp.trials])
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(objective_means.T, axis=1), ylabel=cfg.metric.name,
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    best_objective_plot2 = optimization_trace_single_method(
        y=objective_means.T, ylabel=cfg.metric.name,
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )

    data = best_objective_plot[0]['data']
    lay = best_objective_plot[0]['layout']
    lay['paper_bgcolor'] = 'rgba(0,0,0,0)'
    lay['plot_bgcolor'] = 'rgba(0,0,0,0)'
    fig = {
        "data": best_objective_plot[0]['data'] + best_objective_plot2[0]['data'],  # data,
        "layout": lay,
    }
    import plotly.graph_objects as go
    go.Figure(fig).show()  # write_image("test.pdf")

    render(best_objective_plot)

    from ax.utils.notebook.plotting import render, init_notebook_plotting
    from ax.plot.contour import plot_contour

    plot = plot_contour(model=gpei,
                        param_x="N",
                        param_y="L",
                        metric_name="Strain", )
    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image("test.pdf")

    render(plot)


if __name__ == '__main__':
    sys.exit(mems_exp())
