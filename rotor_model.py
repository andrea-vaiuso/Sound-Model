from scipy.optimize import minimize
import numpy as np

def convert_rpm_to_scaled_radians(rpm):
    return (rpm * 2 * np.pi / 60) / 10

def _calculate_predicted_Lw_total(params, data, num_rotors=4):
    """
    Calculate predicted total sound power levels using the regression model.

    Parameters:
    - params (array-like): Flattened array of a, b, c, d coefficients.
    - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
    - num_rotors (int): Number of rotors (default: 4).

    Returns:
    - predicted_Lw_total (array-like): Predicted total sound power levels.
    """
    # Reshape params to extract coefficients a, b, c, d
    n_frequencies = len(data['Lw_ref'][0])
    a, b, c, d = params[:n_frequencies], params[n_frequencies:2 * n_frequencies], params[2 * n_frequencies:3 * n_frequencies], params[3 * n_frequencies:]

    # Compute predicted Lw_individual for each rotor and then combine them
    predicted_Lw_total = []
    for Lw_ref, zeta, RPMs, C_proc in zip(data['Lw_ref'], data['zeta'], data['RPM'], data['C_proc']):
        Lw_individual_list = [
            Lw_ref + a * (zeta ** 2) + b * np.abs(zeta) + c * RPM + d * (RPM ** 2) + C_proc - 10 * np.log10(num_rotors)
            for RPM in RPMs
        ]
        Lw_total = 10 * np.log10(np.sum([10 ** (Lw / 10) for Lw in Lw_individual_list], axis=0))
        predicted_Lw_total.append(Lw_total)

    return np.array(predicted_Lw_total)

def _regression_loss(params, data, num_rotors=4):
    """
    Compute the loss (mean squared error) between predicted and actual Lw_total.

    Parameters:
    - params (array-like): Flattened array of a, b, c, d coefficients.
    - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
    - num_rotors (int): Number of rotors (default: 4).

    Returns:
    - mse (float): Mean squared error.
    """
    predicted_Lw_total = _calculate_predicted_Lw_total(params, data, num_rotors)
    actual_Lw_total = np.concatenate(data['Lw_total'], axis=0)
    predicted_Lw_total_flat = np.concatenate(predicted_Lw_total, axis=0)  # Flatten to match actual data
    mse = np.mean((predicted_Lw_total_flat - actual_Lw_total) ** 2)
    return mse

def model_fit(data, num_rotors=4, seed_value=42):
    """
    Optimize the coefficients a, b, c, d to minimize the regression loss.

    Parameters:
    - data (dict): Dataset containing Lw_ref, zeta, RPM, C_proc, and actual Lw_total.
    - num_rotors (int): Number of rotors (default: 4).

    Returns:
    - result (OptimizeResult): Result of the optimization.
    """
    n_frequencies = len(data['Lw_ref'][0])
    initial_guess = np.zeros(4 * n_frequencies)  # Initial guess for a, b, c, d coefficients
    #np.random.seed(seed_value)
    #initial_guess = np.random.uniform(low=-1, high=1, size=4 * n_frequencies)

    result = minimize(
        _regression_loss,
        initial_guess,
        args=(data, num_rotors),
        method='L-BFGS-B',
        options={'maxiter': 10000, 'disp': True}
    )

    optimized_params = result.x
    a, b, c, d = (
        optimized_params[:n_frequencies],
        optimized_params[n_frequencies:2 * n_frequencies],
        optimized_params[2 * n_frequencies:3 * n_frequencies],
        optimized_params[3 * n_frequencies:]
    )
    print("Number of iterations:", result.nit)
    print("Final loss (objective function value):", result.fun)
    return a, b, c, d

def execute_model(input, a, b, c, d, num_rotors=4):
    """
    Execute the model using the optimized coefficients a, b, c, d.

    Parameters:
    - input (dict): Dataset containing Lw_ref, zeta, RPM, C_proc.
    - a, b, c, d (array-like): Optimized coefficients.
    - num_rotors (int): Number of rotors (default: 4).

    Returns:
    - predicted_Lw_total (array-like): Predicted total sound power levels.
    """
    n_frequencies = len(input['Lw_ref'][0])
    params = np.concatenate([a, b, c, d])
    predicted_Lw_total = _calculate_predicted_Lw_total(params, input, num_rotors)
    return predicted_Lw_total