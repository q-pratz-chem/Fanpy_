"""Solvers for single equations."""

from fanpy.eqn.base import BaseSchrodinger
from fanpy.eqn.energy_oneside import EnergyOneSideProjection
from fanpy.eqn.energy_twoside import EnergyTwoSideProjection
from fanpy.eqn.energy_variational import EnergyVariational
from fanpy.eqn.least_squares import LeastSquaresEquations
from fanpy.solver.wrappers import wrap_scipy

import numpy as np


def cma(objective, **kwargs):
    """Solve an equation using Covariance Matrix Adaptation Evolution Strategy.

    See module `cma` for details.

    Parameters
    ----------
    objective : BaseSchrodinger
        Instance that contains the function that will be optimized.
    kwargs : dict
        Keyword arguments to `cma.fmin`. See its documentation for details.
        By default, 'sigma0' is set to 0.01 and 'options' to `{'ftarget': None, 'timeout': np.inf,
        'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0}`.
        The 'sigma0' is the initial standard deviation. The optimum is expected to be within
        `3*sigma0` of the initial guess.
        The 'ftarget' is the termination condition for the function value upper limit.
        The 'timeout' is the termination condition for the time. It is provided in seconds and must
        be provided as a string.
        The 'tolfun' is the termination condition for the change in function value.
        The 'verb_filenameprefix' is the prefix of the logger files that will be written to disk.
        The 'verb_log' is the verbosity of the logger files that will be written to disk. `0` means
        that no logs will be saved.
        See `cma.evolution_strategy.cma_default_options` for more options.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy after optimization.
        Only available for objectives that are EnergyOneSideProjection, EnergyTwoSideProjection, and
        LeastSquaresEquations instances.
    message : str
        Termination reason.
    internal : list
        Returned value of the `cma.fmin`.

    Raises
    ------
    TypeError
        If objective is not BaseSchrodinger instance.
    ValueError
        If objective has more than one equation.

    """
    import cma as solver  # pylint: disable=C0415

    if not isinstance(objective, BaseSchrodinger):
        raise TypeError("Objective must be a BaseSchrodinger instance.")
    if objective.num_eqns != 1:
        raise ValueError("Objective must contain only one equation.")

    # disable hamiltonian update because algorithm is stochastic
    objective.ham.update_prev_params = False
    # disable print with each iteration b/c solver prints
    objective.step_print = False

    kwargs.setdefault("sigma0", 0.01)
    kwargs.setdefault("options", {})
    kwargs["options"].setdefault("ftarget", None)
    kwargs["options"].setdefault("timeout", np.inf)
    kwargs["options"].setdefault("tolfun", 1e-11)
    kwargs["options"].setdefault("verb_log", 0)

    if objective.active_params.size == 1:
        raise ValueError("CMA solver cannot be used on objectives with only one parameter.")

    results = solver.fmin(objective.objective, objective.active_params, **kwargs)

    output = {}
    output["success"] = results[-3] != {}
    output["params"] = results[0]
    output["function"] = results[1]

    if isinstance(objective, LeastSquaresEquations):
        output["energy"] = objective.energy.params
    elif isinstance(objective, (EnergyOneSideProjection, EnergyTwoSideProjection)):  # pragma: no branch
        output["energy"] = results[1]

    if output["success"]:  # pragma: no branch
        output["message"] = "Following termination conditions are satisfied:" + "".join(
            " {0}: {1},".format(key, val) for key, val in results[-3].items()
        )
        output["message"] = output["message"][:-1] + "."
    else:  # pragma: no cover
        output["message"] = "Optimization did not succeed."

    output["internal"] = results

    objective.assign_params(results[0])
    objective.save_params()

    return output


def minimize(objective, use_gradient=True, **kwargs):
    """Solve an equation using `scipy.optimize.minimize`.

    See module `scipy.optimize.minimize` for details.

    Parameters
    ----------
    objective : BaseSchrodinger
        Instance that contains the function that will be optimized.
    use_gradient : bool
        Option to use gradient.
        Default is True.
    kwargs : dict
        Keyword arguments to `scipy.optimize.minimize`. See its documentation for details.
        By default, if the objective has a gradient,  'method' is 'BFGS', 'jac' is the gradient
        of the objective, and 'options' is `{'gtol': 1e-8}`.
        By default, if the objective does not have a gradient, 'method' is 'Powell' and 'options' is
        `{'xtol': 1e-9, 'ftol': 1e-9}}`.

    Returns
    -------
    Dictionary with the following keys and values:
    success : bool
        True if optimization succeeded.
    params : np.ndarray
        Parameters at the end of the optimization.
    energy : float
        Energy after optimization.
        Only available for objectives that are EnergyOneSideProjection, EnergyTwoSideProjection, and
        LeastSquaresEquations instances.
    message : str
        Message returned by the optimizer.
    internal : list
        Returned value of the `scipy.optimize.minimize`.

    Raises
    ------
    TypeError
        If objective is not BaseSchrodinger instance.
    ValueError
        If objective has more than one equation.

    """
    import scipy.optimize  # pylint: disable=C0415

    if not isinstance(objective, BaseSchrodinger):
        raise TypeError("Objective must be a BaseSchrodinger instance.")
    if objective.num_eqns != 1:
        raise ValueError("Objective must contain only one equation.")

    if use_gradient:
        kwargs.setdefault("method", "BFGS")
        kwargs.setdefault("jac", objective.gradient)
        kwargs.setdefault("options", {})
        kwargs["options"].setdefault("gtol", 1e-8)
    else:
        kwargs.setdefault("method", "Powell")
        kwargs.setdefault("options", {})
        kwargs["options"].setdefault("xtol", 1e-9)
        kwargs["options"].setdefault("ftol", 1e-9)

    ham = objective.ham
    ham.update_prev_params = False

    def update_iteration(*args):  # pylint: disable=W0613
        """Clean up at the end of each iteration."""
        # update hamiltonian
        if objective.indices_component_params[ham].size > 0:
            ham.update_prev_prams = True
            ham.assign_params(ham.params)
            ham.update_prev_prams = False
        # save parameters
        objective.save_params()
        # print
        for key, value in objective.print_queue.items():
            print("(Mid Optimization) {}: {}".format(key, value))

    kwargs.setdefault("callback", update_iteration)

    output = wrap_scipy(scipy.optimize.minimize)(objective, **kwargs)
    output["function"] = output["internal"].fun
    if isinstance(objective, LeastSquaresEquations):
        output["energy"] = objective.energy.params
    elif isinstance(objective, (EnergyOneSideProjection, EnergyTwoSideProjection)):  # pragma: no branch
        output["energy"] = output["function"]

    return output


def adam(objective, lr=0.001, betas=(0.9,0.99), max_iter=1000, **kwargs):
    """Optimize the given objective using the Adam optimizer. (PyTorch).

    Parameters
    ----------
    objective : BaseSchrodinger
        Instance that contains the function that will be optimized.
    lr : float
        Learning rate.
    max_iter : int
        Maximum number of iterations.
    kwargs : dict
        Extra arguments (e.g., weight_decay, batch_size).

    Returns
    -------
    dict
        Results of the optimization in the same format as cma/minimize.
    """
    import torch
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR

    seed = 12345
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)   # may slow things down
    
    tol_E = 1e-7 # tolerance for change in objective value
    tol_grad = 1e-6 # tolerance for objective gradient norm
    prev_energy = None
    # original_lr = lr

    if not isinstance(objective, BaseSchrodinger):
        raise TypeError("Objective must be a BaseSchrodinger instance.")
    if objective.num_eqns != 1:
        raise ValueError("Objective must contain only one equation.")

    # Disable hamiltonian updates (Adam is iterative)
    objective.ham.update_prev_params = False
    objective.step_print = False

    # Convert parameters to PyTorch tensors
    params = torch.tensor(
        objective.wfn.params, requires_grad=True, dtype=torch.float64
    )
    optimizer = Adam(
        [params],
        lr=lr,
        betas=betas,
        # weight_decay=kwargs.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)  # Reduce LR every 1000 steps by 10%

    # Gradient clipping threshold
    max_grad_norm = kwargs.get("max_grad_norm", 20.0)

    def loss_fn():
        # Convert params to numpy and evaluate Fanpy objective
        params_np = params.detach().cpu().numpy() if isinstance(params, torch.Tensor) else np.array(params)
        # if hasattr(objective.wfn, "normalize"):
        #     objective.wfn.normalize(objective.pspace_l)
        return objective.objective(params_np) #, normalize=True) #, assign=True)
    
    print('###\nPerforming Adam optimization with lr = {}, betas = {}, max_iter = {}'.format(lr, betas, max_iter))
    print(f'Scheduler: Reducing LR every {scheduler.step_size} steps by {(1-scheduler.gamma)*100}%')
    print('Initial parameters: {}'.format(params.detach().numpy()))
    print('Initial loss: {}'.format(loss_fn()))
    print('###')

    for i in range(max_iter):
        optimizer.zero_grad()

        # Compute the loss
        loss_value = loss_fn()

        # Normalize the wavefunction before computing gradients
        if hasattr(objective.wfn, "normalize"):
            # print("objective.pspace_n: ", objective.pspace_n)
            objective.wfn.normalize(objective.pspace_n)

        grad_np = objective.gradient(params.detach().numpy())
        # Gradient clipping
        grad_norm = np.linalg.norm(grad_np)
        print(f"Iteration {i}, Loss = {loss_value}, Grad norm = {grad_norm}")
        if grad_norm > max_grad_norm:
            grad_np = grad_np * (max_grad_norm / grad_norm)
            print("Clipped gradient norm from {} to {}".format(grad_norm, max_grad_norm))

        params.grad = torch.from_numpy(grad_np).to(params.device, dtype=torch.float64)

        # Backpropogation
        # loss.backward()
        # print(f"Gradients: {params.grad}")

        # Check convergence
        if prev_energy is not None:
            delta_E = abs(loss_value - prev_energy)
            grad_norm = np.linalg.norm(grad_np)
            if delta_E < tol_E and grad_norm < tol_grad:
                print(f"Iteration {i}, Loss = {loss_value}")
                print(f"Converged at iteration {i}, ΔE={delta_E}, ||grad||={grad_norm}")
                print(f'Final parameters: {params.detach().numpy()}')
                break
        prev_energy = loss_value

        # Optimization step
        optimizer.step()
        scheduler.step() # Update learning rate

        # Get the current learning rate from the scheduler
        # current_lr = scheduler.get_last_lr()
        # if original_lr != current_lr[0]:
            # print(f"Iteration {i}, Learning Rate: {current_lr}")


        # Update Fanpy parameters   
        # print(f"Updated parameters: {params.detach().numpy()}")
        objective.wfn.assign_params(params.detach().numpy())

        # Renormalize the wavefunction after updating parameters
        # if hasattr(objective.wfn, "normalize"):
        #     objective.wfn.normalize(objective.pspace_n)

        # if i % 10 == 0:
        #     print(f"Iteration {i}, Loss = {loss_value}")

    output = {
        "success": True,
        "params": params.detach().numpy(),
        "energy": loss_fn(),
        "message": "Optimization completed with Adam.",
        "internal": None,
    }
    return output

