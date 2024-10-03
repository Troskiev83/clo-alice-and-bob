import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.converters import QuadraticProgramToQubo



def run(input_data, solver_params, extra_arguments):
    # Extract data from input
    loans = input_data["loans"]
    industry_concentration_limit = input_data["industry_concentration_limit"]
    country_concentration_limit = input_data["country_concentration_limit"]
    credit_risk_threshold = input_data["credit_risk_threshold"]
    excluded_sectors = input_data["excluded_sectors"]
    total_exposure_limit = input_data["total_exposure_limit"]

    # Number of loans
    n_loans = len(loans)

    # Initialize the Quadratic Program
    qp = QuadraticProgram()

    # Add binary decision variables x_i
    for i in range(n_loans):
        qp.binary_var(name=f"x_{i}")

    # Penalty coefficient
    penalty = 1e6  # Adjust as necessary

    # Objective function: maximize expected returns (minimize negative returns)
    linear = {f"x_{i}": -loan['expected_return'] for i, loan in enumerate(loans)}
    quadratic = {}

    # Total possible exposure (sum of exposures of all loans)
    total_possible_exposure = sum(loan['exposure'] for loan in loans)

    # Convert constraints to penalty terms and add to the objective function

    # Industry concentration limits
    for j in range(len(loans[0]['industries'])):
        coeffs = [loan['exposure'] * loan['industries'][j] for loan in loans]
        limit = industry_concentration_limit * total_possible_exposure

        # Build the penalty term: (sum_i coeffs_i * x_i - limit)^2
        for i in range(n_loans):
            xi = f"x_{i}"
            ci = coeffs[i]
            # Linear terms
            linear[xi] = linear.get(xi, 0) + penalty * (-2 * ci * limit)
            # Quadratic terms
            for j2 in range(i, n_loans):
                xj = f"x_{j2}"
                cj = coeffs[j2]
                key = (xi, xj)
                quadratic[key] = quadratic.get(key, 0) + penalty * ci * cj
        # Add constant term
        qp.objective.constant += penalty * limit ** 2

    # Country concentration limits
    for k in range(len(loans[0]['countries'])):
        coeffs = [loan['exposure'] * loan['countries'][k] for loan in loans]
        limit = country_concentration_limit * total_possible_exposure

        for i in range(n_loans):
            xi = f"x_{i}"
            ci = coeffs[i]
            linear[xi] = linear.get(xi, 0) + penalty * (-2 * ci * limit)
            for j2 in range(i, n_loans):
                xj = f"x_{j2}"
                cj = coeffs[j2]
                key = (xi, xj)
                quadratic[key] = quadratic.get(key, 0) + penalty * ci * cj
        qp.objective.constant += penalty * limit ** 2

    # Excluded sectors
    for l in excluded_sectors:
        coeffs = [loan['sectors'][l] if l < len(loan['sectors']) else 0 for loan in loans]
        for i in range(n_loans):
            xi = f"x_{i}"
            ci = coeffs[i]
            # No linear term since RHS is 0
            for j2 in range(i, n_loans):
                xj = f"x_{j2}"
                cj = coeffs[j2]
                key = (xi, xj)
                quadratic[key] = quadratic.get(key, 0) + penalty * ci * cj
        # No constant term needed

    # Credit risk threshold constraint
    for i, loan in enumerate(loans):
        if loan['credit_risk_rating'] >= credit_risk_threshold:
            xi = f"x_{i}"
            # Penalty term: penalty * x_i
            linear[xi] = linear.get(xi, 0) + penalty

    # Total exposure constraint
    if total_exposure_limit:
        coeffs = [loan['exposure'] for loan in loans]
        limit = total_exposure_limit
        for i in range(n_loans):
            xi = f"x_{i}"
            ci = coeffs[i]
            linear[xi] = linear.get(xi, 0) + penalty * (-2 * ci * limit)
            for j2 in range(i, n_loans):
                xj = f"x_{j2}"
                cj = coeffs[j2]
                key = (xi, xj)
                quadratic[key] = quadratic.get(key, 0) + penalty * ci * cj
        qp.objective.constant += penalty * limit ** 2

    # Update the objective function
    qp.objective.linear = linear
    qp.objective.quadratic = quadratic

    # Initialize the converter
    qubo_converter = QuadraticProgramToQubo()

    # Convert the Quadratic Program to QUBO
    try:
        qubo = qubo_converter.convert(qp)
    except Exception as e:
        res = {
            "selected_loans": [0]*n_loans,
            "total_expected_return": 0,
            "total_exposure": 0,
            "success": False,
            "error": str(e)
        }
        return res

    # Set up QAOA with the Alice and Bob simulator
    p = solver_params.get('p', 1)
    optimizer = COBYLA(maxiter=solver_params.get('maxiter', 100))

    # Initialize the sampler
    sampler = Sampler()

    qaoa = QAOA(sampler=sampler, reps=p, optimizer=optimizer)

    # Use MinimumEigenOptimizer with QAOA
    eigen_optimizer = MinimumEigenOptimizer(qaoa)

    result = eigen_optimizer.solve(qubo)

    # Extract results
    selected_loans = [int(result.x[i]) for i in range(n_loans)]
    total_expected_return = -result.fval  # Since we minimized the negative expected return
    total_exposure = sum(loans[i]['exposure'] for i in range(n_loans) if selected_loans[i])

    res = {
        "selected_loans": selected_loans,
        "total_expected_return": total_expected_return,
        "total_exposure": total_exposure,
        "success": result.status.name == 'SUCCESS'
    }

    return res
