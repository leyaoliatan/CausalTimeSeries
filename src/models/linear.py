from sklearn.linear_model import LinearRegression
import numpy as np


def fit_ar1(df, treated_unit):
    """Fit AR(1) model and generate predictions for post-treatment periods
    """
    # Get pre-treatment data for the treated unit
    treated_pre = df[(df['subject_id'] == treated_unit) & (df['is_post'] == False)]
    treated_pre_Y = treated_pre['Y'].values
    n_forecast_periods = len(df[df['is_post'] == True]['time'].unique())
    
    # Create lagged data for AR(1): y(t) = β₀ + β₁y(t-1) + ε(t)
    Y_lag = treated_pre_Y[:-1].reshape(-1, 1)    # y(t-1)
    Y_target = treated_pre_Y[1:]                 # y(t)
    
    # Fit AR(1) model using linear regression
    ar_model = LinearRegression(fit_intercept=True)  # Important: include intercept
    ar_model.fit(Y_lag, Y_target)
    
    # Generate predictions iteratively
    predictions = []
    last_observed = treated_pre_Y[-1]  # Use last pre-treatment value
    
    for _ in range(n_forecast_periods):
        # Predict next value: y(t) = β₀ + β₁y(t-1)
        next_prediction = ar_model.predict([[last_observed]])[0]
        predictions.append(next_prediction)
        last_observed = next_prediction  # Use prediction as next lag
    
    return np.array(predictions)
