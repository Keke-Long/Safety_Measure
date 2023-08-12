import pandas as pd
import numpy as np
from scipy.optimize import minimize


lane_num = 1
PET_threshold = 5
# Reload the dataset
data = pd.read_csv(f"../Data/HighSIM/HighSim_safety_measures_lane{lane_num}.csv")

# Get the unique car ids
car_ids = data['id'].unique()

# Initialize a dictionary to store the risk value for each car
risk_result = {}


# Define the negative log likelihood function for GPD
def neg_log_likelihood(params, data):
    k, sigma = params
    if sigma <= 0 or k >= 0.5:
        return np.inf
    else:
        return -np.sum(np.log((1/k) * (1 + k * data / sigma) ** (-(1/k + 1))))


init_params = [0.1, 1] # Initial guess for k and sigma

# Process each car
for car_id in car_ids:
    car_data = data[data['id'] == car_id]

    if (car_data['PET'] < PET_threshold).any():
        # Compute -PET for the values less than the threshold
        negative_pet = -car_data.loc[car_data['PET'] < PET_threshold, 'PET']
        negative_pet_array = negative_pet.values

        # Fit a GPD to the -PET values
        result = minimize(neg_log_likelihood, init_params, args=(negative_pet_array,), method='Nelder-Mead')

        # If the optimization was successful, compute the GPD value at 0
        if result.success:
            k_hat, sigma_hat = result.x
            if k_hat == 0:
                risk_value = 1 / sigma_hat
            else:
                risk_value = 1 / sigma_hat * (1 + k_hat * 0 / sigma_hat) ** (-(1/k_hat + 1))
        else:
            # If the optimization was not successful, set the risk value to NaN
            risk_value = np.nan
    else:
        # If there are no PET values less than the threshold, set the risk value to 999
        risk_value = np.nan

    # Store the risk value for this car
    risk_result[car_id] = round(risk_value,5)

# Convert the risk values dictionary to a dataframe
risk_df = pd.DataFrame.from_dict(risk_result, orient='index', columns=['Risk'])

risk_df.reset_index(inplace=True)
risk_df.rename(columns={'index': 'id'}, inplace=True)

risk_df.to_csv(f"../Data/HighSIM/risk_values_lane{lane_num}.csv", index=False)
n1 = risk_df['Risk'].count()
n2 = risk_df['id'].count()
print(f'成功计算出{n1}辆车的risk值，占比{100*n1/n2}%')