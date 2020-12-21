import plotly.graph_objects as go
import numpy as np
from ax.plot.trace import optimization_trace_single_method
import os


def save_fig(plot, dir):
    data = plot[0]['data']
    lay = plot[0]['layout']

    fig = {
        "data": data,
        "layout": lay,
    }
    fig = go.Figure(fig)
    fig.update_layout(
        font_family="Times New Roman",
        font_color="Black",
        font_size=14,
        margin=dict(r=5,t=10,l=20,b=20)
    )
    fig.write_image(os.getcwd() + "/" + dir  + '.pdf')


def plot_learning(exp, cfg):
    objective_means = np.array([[exp.trials[trial].objective_mean] for trial in exp.trials])
    cumulative = optimization_trace_single_method(
        y=np.maximum.accumulate(objective_means.T, axis=1) * 1.01, ylabel=cfg.metric.name,
        trace_color=tuple((83, 78, 194)),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )
    all = optimization_trace_single_method(
        y=objective_means.T, ylabel=cfg.metric.name,
        model_transitions=[cfg.bo.random], trace_color=tuple((114, 110, 180)),
        # optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
    )

    layout_learn = cumulative[0]['layout']
    layout_learn['paper_bgcolor'] = 'rgba(0,0,0,0)'
    layout_learn['plot_bgcolor'] = 'rgba(0,0,0,0)'
    layout_learn['showlegend'] = False

    d1 = cumulative[0]['data']
    d2 = all[0]['data']

    for t in d1:
        t['legendgroup'] = cfg.metric.name + ", cum. max"
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name + ", cum. max"
            t['line']['color'] = 'rgba(200,20,20,0.5)'
            t['line']['width'] = 4

    for t in d2:
        t['legendgroup'] = cfg.metric.name
        if 'name' in t and t['name'] == 'Generator change':
            t['name'] = 'End Random Iterations'
        else:
            t['name'] = cfg.metric.name
            t['line']['color'] = 'rgba(20,20,200,0.5)'
            t['line']['width'] = 4

    fig = {
        "data": d1 + d2,  # data,
        "layout": layout_learn,
    }
    return fig


# Adapted from Gpy

# ===============================================================================
# Copyright (c) 2015, Max Zwiessele
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of GPy.plotting.gpy_plot.kernel_plots nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================
def plot_covariance(kernel, x=None, label=None,
                    plot_limits=None, visible_dims=None, resolution=None,
                    projection='2d', levels=20, **kwargs):
    """
    Plot a kernel covariance w.r.t. another x.
    :param array-like x: the value to use for the other kernel argument (kernels are a function of two variables!)
    :param plot_limits: the range over which to plot the kernel
    :type plot_limits: Either (xmin, xmax) for 1D or (xmin, xmax, ymin, ymax) / ((xmin, xmax), (ymin, ymax)) for 2D
    :param array-like visible_dims: input dimensions (!) to use for x. Make sure to select 2 or less dimensions to plot.
    :resolution: the resolution of the lines used in plotting. for 2D this defines the grid for kernel evaluation.
    :param {2d|3d} projection: What projection shall we use to plot the kernel?
    :param int levels: for 2D projection, how many levels for the contour plot to use?
    :param kwargs:  valid kwargs for your specific plotting library
    """
    X = np.ones((2, kernel._effective_input_dim)) * [[-3], [3]]
    _, free_dims, Xgrid, xx, yy, _, _, resolution = helper_for_plot_data(kernel, X, plot_limits, visible_dims, None,
                                                                         resolution)

    from numbers import Number
    if x is None:
        from ...kern.src.stationary import Stationary
        x = np.ones((1, kernel._effective_input_dim)) * (not isinstance(kernel, Stationary))
    elif isinstance(x, Number):
        x = np.ones((1, kernel._effective_input_dim)) * x
    K = kernel.K(Xgrid, x)

    if projection == '3d':
        xlabel = 'X[:,0]'
        ylabel = 'X[:,1]'
        zlabel = "k(X, {!s})".format(np.asanyarray(x).tolist())
    else:
        xlabel = 'X'
        ylabel = "k(X, {!s})".format(np.asanyarray(x).tolist())
        zlabel = None

    canvas, kwargs = pl().new_canvas(projection=projection, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)

    if len(free_dims) <= 2:
        if len(free_dims) == 1:
            # 1D plotting:
            update_not_existing_kwargs(kwargs, pl().defaults.meanplot_1d)  # @UndefinedVariable
            plots = dict(covariance=[pl().plot(canvas, Xgrid[:, free_dims], K, label=label, **kwargs)])
        else:
            if projection == '2d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_2d)  # @UndefinedVariable
                plots = dict(covariance=[pl().contour(canvas, xx[:, 0], yy[0, :],
                                                      K.reshape(resolution, resolution),
                                                      levels=levels, label=label, **kwargs)])
            elif projection == '3d':
                update_not_existing_kwargs(kwargs, pl().defaults.meanplot_3d)  # @UndefinedVariable
                plots = dict(covariance=[pl().surface(canvas, xx, yy,
                                                      K.reshape(resolution, resolution),
                                                      label=label,
                                                      **kwargs)])
        return pl().add_to_canvas(canvas, plots)

    else:
        raise NotImplementedError("Cannot plot a kernel with more than two input dimensions")
