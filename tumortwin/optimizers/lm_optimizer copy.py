from datetime import datetime
from typing import Any, Callable, List, Optional

import torch


class LMoptions:
    """
    Configuration options for the Levenberg-Marquardt optimization algorithm.

    Attributes:
        jac_delta (float): Step size used to compute the Jacobian via finite differences. Default is 1e-4.
        jac_update_interval (int): Number of accepted steps between updates of the Jacobian. Default is 1.
        bad_run_jac_interval_weight (float): Weight applied to the Jacobian update interval when a bad step occurs. Default is 0.25.
        lambda_init (float): Initial value for the lambda parameter, dictating the balance between gradient descent and Gauss-Newton. Default is 1.
        lambda_upscale_factor (float): Factor to increase the lambda parameter when a bad step occurs, making behavior more similar to gradient descent. Default is 11.
        lambda_downscale_factor (float): Factor to decrease the lambda parameter when a good step occurs, making behavior more similar to Gauss-Newton. Default is 9.
        max_initial_delta (float): Maximum initial step size relative to parameter bounds. Default is 0.1.
        jac_function (Optional[Callable]): Custom function to compute the Jacobian. If None, the Jacobian is computed numerically. Default is None.
        accept_thresh (float): The threshold for accepting a step based on the error reduction.
    """

    def __init__(
        self,
        jac_delta: float = 1e-4,
        jac_update_interval: int = 1,
        bad_run_jac_interval_weight: float = 0.25,
        lambda_init: float = 1.0,
        lambda_upscale_factor: float = 5.0,
        lambda_downscale_factor: float = 1.5,
        max_initial_delta: float = 0.1,
        jac_function: Optional[Callable] = None,
        accept_thresh: float = 0.0,
    ):

        self.jac_delta = torch.tensor(jac_delta)
        self.jac_update_interval = jac_update_interval
        self.bad_run_jac_interval_weight = bad_run_jac_interval_weight
        self.lambda_init = torch.tensor(lambda_init)
        self.lambda_upscale_factor = torch.tensor(lambda_upscale_factor)
        self.lambda_downscale_factor = torch.tensor(lambda_downscale_factor)
        self.max_initial_delta = torch.tensor(max_initial_delta)
        self.jac_function = jac_function
        self.accept_thresh = torch.tensor(accept_thresh)


class LMoptimizer:
    """
    A Levenberg-Marquardt optimizer for nonlinear least squares problems.

    This class implements the Levenberg-Marquardt optimization algorithm to solve problems
    of the form min_x || model(x) - y_data ||^2, where model(x) is a nonlinear function and y_data is
    the observed data. It combines gradient descent and Gauss-Newton methods to adaptively
    optimize the model parameters x.

    Attributes:
        model (Callable): The nonlinear function to optimize, which should return a vector.
        bounds (torch.Tensor): The bounds for each parameter of the optimization.
        x (torch.Tensor): The current guess for the parameters.
        initial_guess (torch.Tensor): The initial guess for the parameters.
        best_x (torch.Tensor): The best found parameters during the optimization.
        y_data (torch.Tensor): The observed data.
        _lambda (float): The damping factor for the Levenberg-Marquardt update.
        options: (LMOptions): Optimizer parameters (see LMOptions)
        parameters (List[torch.Tensor]): List of best parameters at each optimization iteration
        error_record (List[float]): List of error at each optimization iteration
        error (List[float]): List of best error at each optimization iteration

    Methods:
        step(executor: Optional[Callable] = None) -> None:
            Performs one optimization step, adjusting the parameters based on the current Jacobian and error.
    """

    def __init__(
        self,
        solver: Any,
        timepoints: List[datetime],
        u0: torch.Tensor,
        bounds: torch.Tensor,
        initial_guess: torch.Tensor,
        y_data: torch.Tensor,
        options: LMoptions = LMoptions(),
    ):
        """
        Initializes the Levenberg-Marquardt optimizer with given parameters.

        Args:
            model (Callable): The function to be optimized. Takes an input array and returns an output array.
            bounds (torch.Tensor): Bounds to constrain the input guesses. Each element should be a (lower bound, upper bound) tuple. Use empty tuples for unbounded inputs.
            initial_guess (torch.Tensor): Initial guess for the input.
            y_data (torch.Tensor): Observed data for comparison with the function output. Levenberg-Marquardt will attempt to match to this data.
            options (LMoptionsTorch): Configuration options for the Levenberg-Marquardt optimization algorithm.

        """
        
        self.solver = solver
        self.timepoints = timepoints
        self.u0 = u0
        self.bounds = bounds
        self.x = initial_guess.requires_grad_(True)
        self.x0 = self.x.requires_grad_(True)
        self.best_x = self.x.requires_grad_(True)
        self.y_data = y_data.reshape(-1)
        self.options = options
        self._lambda = options.lambda_init

        # Internal variables
        self._n_function_calls: int = 0
        self._n_iters: int = 0
        self._stuck_count: int = 0
        self._jac_update: float = 0.0
        self._y_next_error: torch.Tensor = torch.zeros_like(self.y_data)
        self._accepted_step: bool = False
        self._best_y: Optional[torch.Tensor] = None
        self.parameters: List[torch.Tensor] = []
        self.error_record: List[float] = []
        self.error: List[float] = []

        self._bound_inputs(self.x, throw_error=True)

    def _update_params(self, x : torch.Tensor):
        d, k, ct_sens = x
        self.solver.model.d = torch.nn.Parameter(d)
        self.solver.model.k = torch.nn.Parameter(k)
        self.solver.model.chemotherapy_specifications[0].sensitivity = torch.nn.Parameter(ct_sens)
    
    def model(self):
        _, predicted_cellularity_maps = self.solver.solve(timepoints=self.timepoints, u_initial=self.u0)
        return predicted_cellularity_maps
        
    def _bound_inputs(self, x: torch.Tensor, throw_error: bool = False) -> None:
        """
        Ensures that each parameter is within provided bounds.

        Args:
            x (torch.Tensor): The current parameter vector.
            throw_error (bool): Whether to raise an exception if a parameter is out of bounds.
        """
        for i, bound in enumerate(self.bounds):
            if len(bound) < 2:
                # no bounds (or only one ambiguous bound) provided for this variable
                continue
            lb, ub = bound[0], bound[1]

            if throw_error:
                if x[i] < lb:
                    raise ValueError(
                        f"Parameter at index {i} is out of bounds: {x[i]} < {lb}"
                    )
                elif x[i] > ub:
                    raise ValueError(
                        f"Parameter at index {i} is out of bounds: {x[i]} > {ub}"
                    )

            # x[i] = torch.clamp(x[i], min=lb, max=ub)

    def step(self) -> None:
        """Executes a single iteration of the Levenberg-Marquardt optimization process."""
        if self._n_iters == 0:
            print("Initial step")
            self._initialize_first_iteration()
        elif self._accepted_step:
            print("Accepted step")
            self._perform_accepted_step()
        else:
            print("Stuck step")
            self._handle_stuck_condition()

        # Perform optimization steps
        self._perform_optimization_step()

        # After iteration, accept step if improvement is significant
        self._accept_step_if_improved()

        # Storing current best solution
        self.parameters.append(self.best_x)
        self.error.append(self._y_error_best.item())

    def _initialize_first_iteration(self) -> None:
        """Initializes the optimization for the first iteration, including the Jacobian and error."""
        self._update_params(self.x)
        self._y_next = self.model()
        # self._y_next = self.model(self.x)
        self._best_y = self._y_next
        self.y = self._y_next.reshape(-1)
        print("Computing jacobians at initial point")
        # self.J = self._get_jacobian()
        self.Jtorch = self._get_jacobian_torch()
        raise ValueError
        self._y_error_best = self.get_error(self.y, self.y_data)
        self.error_record.append(self._y_error_best.item())
        self._n_function_calls += self.x.size(0) + 1
        self.H = self.J.T @ self.J

    def _perform_accepted_step(self) -> None:
        """Performs an accepted optimization step, updating internal variables."""
        self._accepted_step = False
        self._jac_update += 1.0

        self.y = self._y_next.reshape(-1)
        self._best_y = self._y_next
        self.x = self._x_next

        if self._jac_update >= self.options.jac_update_interval:
            self.J = self._get_jacobian()
            self._n_function_calls += self.x.size(0)
            self._jac_update = 0.0

        self._y_error_best = self._y_next_error
        self._lambda /= self.options.lambda_downscale_factor
        self.H = self.J.T @ self.J
        self._stuck_count = 0

    def _handle_stuck_condition(self) -> None:
        """Handles the case when the optimization appears to be stuck, adjusting the damping."""
        self._stuck_count += 1
        self._jac_update += self.options.bad_run_jac_interval_weight
        self._lambda *= self.options.lambda_upscale_factor

    def _perform_optimization_step(self) -> None:
        """Computes the optimization step based on the current Jacobian and error."""

        def _is_any_not_finite(x: torch.Tensor):
            return torch.any(~torch.isfinite(x))

        while True:
            if (
                _is_any_not_finite(self.y_data)
                or _is_any_not_finite(self.y)
                or _is_any_not_finite(self.J)
                or _is_any_not_finite(self.H)
            ):
                raise ValueError("Matrices contain NaN values.")

            self.J = self.J.double()
            self.H = self.H.double()
            lhs_lstsq = self.H + self._lambda.double() * torch.diag(torch.diag(self.H))
            rhs_lstsq = self.J.T @ (self.y_data - self.y)
            self._deltas: torch.Tensor = torch.linalg.lstsq(
                lhs_lstsq, rhs_lstsq
            ).solution  # the solution should be a torch.Tensor

            relative_delta = torch.max(
                torch.abs(self._deltas) / (self.bounds[:, 1] - self.bounds[:, 0])
            )

            if self._n_iters != 0 or relative_delta <= self.options.max_initial_delta:
                break
            self._lambda *= self.options.lambda_upscale_factor

    def _accept_step_if_improved(self) -> None:
        """
        Accepts the step if the error has improved significantly.

        If the error has reduced sufficiently compared to the best error, updates the best parameters
        and accepts the step.
        """
        self._x_next = self.x + self._deltas
        self._bound_inputs(self._x_next)
        self._y_next = self.model(self._x_next).reshape(-1)
        self._n_function_calls += 1
        self._y_next_error = self.get_error(self._y_next, self.y_data)
        self.error_record.append(self._y_next_error.item())
        self._n_iters += 1

        if self._metric() > self.options.accept_thresh:
            self._best_y = self._y_next
            self.best_x = self._x_next
            self._accepted_step = True
            self._y_error_best = self._y_next_error

    # NB: torch jacobians not working currently, to-be-revisited
    def _get_jacobian_torch(self) -> torch.Tensor:
        """Computes the Jacobian matrix of function model at point x using autograd."""
        # Use autograd to compute the Jacobian
        print("self.x requires grad?")
        print(self.x.requires_grad)              # Should be True
        print(torch.is_grad_enabled())      # Should be True
        def model_function(x):
            ret = self.model(x).reshape(-1)
            print("ret requires grad?")
            print(ret.requires_grad)
            print(f"max model {torch.max(ret)}")
            return ret

        # x = self.x.clone().detach().requires_grad_(True)
        x = self.solver.model.k
        y = self.model().sum()
        y.backward()
        print("x.grad is None?", x.grad is None)  # True or False?
        print(self.solver.model.k)
        print(self.solver.k)
        print("y depends on x?", self.solver.k in y.grad_fn.next_functions)  # Optional but useful

        print("output_sum_grad")
        print(self.x.grad)

        jacobian = torch.autograd.functional.jacobian(model_function, self.x, strategy="reverse-mode")
        print(f"shape of torchJacrev: {jacobian.shape}")
        print(f"mean of torchJacrev: {torch.mean(jacobian)}")
        print(f"max of torchJacrev: {torch.max(jacobian)}")
        jacobian = torch.autograd.functional.jacobian(model_function, self.x, strategy="forward-mode",vectorize=True)
        print(f"shape of torchJacfor: {jacobian.shape}")
        print(f"mean of torchJacfor: {torch.mean(jacobian)}")

        return jacobian

    def _get_jacobian(self) -> torch.Tensor:
        """Computes the Jacobian matrix of function model at point self.x."""
        if self.options.jac_function:
            return self.options.jac_function(self.x)
        else:
            jac = torch.stack(
                [self._jacobian_single_var(i) for i in range(self.x.size(0))]
            ).T
            print(f"shape of FDjac: {jac.shape}")
            print(f"mean of FDjac: {torch.mean(jac)}")
            return jac

    def _jacobian_single_var(self, i: int) -> torch.Tensor:
        """
        Computes the partial derivative for a single variable using finite differences.

        Parameters:
            i (int): Index of the input to perturb.

        Returns:
            torch.Tensor: Partial derivatives with respect to the i-th input.
        """
        x = self.x.clone()
        delta = self.options.jac_delta
        x[i] += delta
        deltaY = self.model(x).reshape(-1)
        return torch.tensor((deltaY - self.y) / delta, dtype=torch.float64)

    def _metric(self) -> torch.Tensor:
        """Computes the metric for deciding whether to accept a step based on the error reduction."""
        return self._y_error_best - self._y_next_error

    def get_error(self, y: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        """
        Computes the squared error between the predicted and observed data.

        Args:
            y (torch.Tensor): Predicted data.
            y_data (torch.Tensor): Target output.

        Returns:
            e (float): Squared error between the predicted and observed data.
        """
        solution_deltas = y_data - y
        e = solution_deltas.T @ solution_deltas
        return e
