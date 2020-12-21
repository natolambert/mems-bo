from ax import (
    SimpleExperiment, Models, Runner, OptimizationConfig, Objective,
    SearchSpace
)


@hydra.main(config_path='conf/conf.yaml')
def mems_exp(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    raise NotImplementedError("TODO load experimental data and verify likelihood of differnet points")
    search_space = gen_search_space(cfg.problem)
    outcome_con = gen_outcome_constraints(cfg.problem)

    exp = SimpleExperiment(
        name=cfg.problem.name,
        search_space=SearchSpace(search_space),
        evaluation_function=jumper,
        objective_name="Energy_(uJ)",
        minimize=False,
        outcome_constraints=outcome_con,
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=MEMsMetric(name="Energy_(uJ)"),
            minimize=False,
        ),
    )

    class MyRunner(Runner):
        def run(self, trial):
            return {"name": str(trial.index)}

    exp.runner = MyRunner()
    exp.optimization_config = optimization_config
    from ax.utils.notebook.plotting import render, init_notebook_plotting
    from ax.plot.contour import plot_contour

    print(f"Running {cfg.bo.random} Sobol initialization trials...")
    sobol = Models.SOBOL(exp.search_space)
    num_search = cfg.bo.random
    for i in range(num_search):
        exp.new_trial(generator_run=sobol.gen(1))
        exp.trials[len(exp.trials) - 1].run()

    # data = exp.fetch_data()

    num_opt = cfg.bo.optimized
    for i in range(num_opt):
        if (i % 5) == 0 and cfg.plot_during:
            plot = plot_contour(model=gpei,
                                param_x="N",
                                param_y="L",
                                metric_name="Energy_(uJ)", )
            data = plot[0]['data']
            lay = plot[0]['layout']

            render(plot)

        print(f"Running GP+EI optimization trial {i + 1}/{num_opt}...")
        # Reinitialize GP+EI model at each step with updated data.
        batch = exp.new_trial(generator_run=gpei.gen(1))
        gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

    from ax.models.torch.botorch_defaults import predict_from_model
    import torch
    X = torch.Tensor([[2, 7e-4, 1e-4], [1, 5e-4, 1e-4]]).double()
    mean, cov = predict_from_model(gpei.model.model, X)
    # X(Tensor) – n x d parameters

    ll = log_likelihood(X, mean, cov)
    plot_ll(ll)


if __name__ == '__main__':
    sys.exit(mems_exp())


gpei = Models.BOTORCH(experiment=exp, data=exp.eval())

from ax.models.torch.botorch_defaults import predict_from_model
import torch
X = torch.Tensor([[2, 7e-4, 1e-4], [1, 5e-4, 1e-4]]).double()
mean, cov = predict_from_model(gpei.model.model, X)
# X(Tensor) – n x d parameters

ll = log_likelihood(X, mean, cov)
plot_ll(ll)
