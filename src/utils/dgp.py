import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def gen_temporal_market_data(n_subjects, n_timesteps, d, base_fn, tau_fn, sigma, ar_coef=0.5, n_competitors=4, competitor_correlation=0.3,increase_trend=0, divergence=0):
    X = np.zeros((n_subjects, n_timesteps, d))
    
    for t in range(n_timesteps):
        if t == 0:
            X[:, t, :] = np.random.uniform(0, 1, size=(n_subjects, d))
        else:
            # AR(1)
            X[:, t, :] = (ar_coef * X[:, t-1, :] + 
                         (1-ar_coef) * np.random.uniform(0, 1, size=(n_subjects, d)))
    
    # Generate treatment (only one unit is treated)
    T = np.zeros((n_subjects, n_timesteps))
    treated_unit = 0
    #treated_unit = np.random.randint(0, n_subjects)  # randomly select one unit to treat
    treatment_start = round(n_timesteps // 1.5)
    T[treated_unit, treatment_start:] = 1  # treat after treatment_start

    #randomly select competitors
    competitor_idx = np.random.choice(np.arange(n_subjects)[np.arange(n_subjects) != treated_unit], n_competitors, replace=False)
    
    # Generate outcomes over time
    Y = np.zeros((n_subjects, n_timesteps))
    Y_c = np.zeros((n_subjects, n_timesteps)) #true counterfactual outcomes
    
    for t in range(n_timesteps):
        noise = sigma * np.random.normal(0, 1, size=n_subjects)
        base_outcome = base_fn(X[:, t, :])

        if t == 0:
            Y[:, t] = base_outcome + noise
            Y_c[:, t] = base_outcome + noise  # Initial counterfactual is the same
        else:
            if divergence == 0:
                Y[:, t] = ar_coef * Y[:, t-1] + base_outcome + noise
                Y[:, t] += increase_trend * t  # Add linear trend

                Y_c[:, t] = ar_coef * Y_c[:, t-1] + base_outcome + noise  # Independent counterfactual dynamics
                Y_c[:, t] += increase_trend * t  # Linear trend for counterfactual
            else:
                #hetergenous increase trend 
                Y[:, t] = ar_coef * Y[:, t-1] + base_outcome + noise
                Y[:, t] +=  increase_trend * t  
                Y[treated_unit, t] += divergence * t

                Y_c[:, t] = ar_coef * Y_c[:, t-1] + base_outcome + noise  
                Y_c[:, t] += increase_trend * t  
                Y_c[treated_unit, t] += divergence * t

            # Competitor effect (only in observed outcomes)
            competitor_mean = Y[competitor_idx, t].mean()
            treated_baseline = Y[treated_unit, t-1]
            #competitor_effect = competitor_correlation * (competitor_mean / (treated_baseline + 1e-6))
            competitor_effect = competitor_correlation * (competitor_mean)
            base = Y[treated_unit, t]
            Y[treated_unit, t] = competitor_effect + (1 - competitor_correlation) * base
            base = Y_c[treated_unit, t]
            Y_c[treated_unit, t] = competitor_effect + (1 - competitor_correlation) * base

        # Add treatment effect (only in observed outcomes)
        if t >= treatment_start:
            treatment_effect = tau_fn(X[treated_unit, t, :])
            Y[treated_unit, t] += treatment_effect
    
    return Y, T, X, treated_unit, competitor_idx, Y_c

def get_market_data_generator(n_subjects, n_timesteps, d, sigma, n_competitors, competitor_correlation=0.3, setup = 'A', increase_trend=0,divergence=0):
    if setup == 'A':
        def base_fn(X): 
            return (0.5 * X[:, 0] + 0.3 * X[:, 1] + 
                   0.2 * X[:, 2] + 0.4 * X[:, 3] + 0.1 * X[:, 4])
        def tau_fn(X):
            # Treatment effect (heterogeneous)
            return 3 + np.sin(2 * np.pi * X[0])
    elif setup == 'B':
        def base_fn(X): 
            # Simple linear combination of features
            return (0.5 * X[:, 0] + 0.3 * X[:, 1] + 
                   0.2 * X[:, 2] + 0.4 * X[:, 3] + 0.1 * X[:, 4])
        def tau_fn(X):
            # constant treatment effect
            return 3.0
    elif setup == 'C':
        def base_fn(X): 
            return (0.5 * X[:, 0] + 0.3 * X[:, 1] + 
                   0.2 * X[:, 2] + 0.4 * X[:, 3] + 0.1 * X[:, 4])
        def tau_fn(X):
            # Treatment effect (heterogeneous)
            return 3*X[0]
    
    def gen_data_fn(): 
        return gen_temporal_market_data(n_subjects, n_timesteps, d, 
                                      base_fn, tau_fn, sigma, 
                                      n_competitors=n_competitors, 
                                      competitor_correlation=competitor_correlation, 
                                      increase_trend=increase_trend, divergence=divergence)
    
    return gen_data_fn, base_fn, tau_fn


def plot_tau_function(tau_fn, n_points=100, figsize=(8, 6)):
    # Create X values
    X = np.linspace(0, 1, n_points)
    
    # Calculate treatment effects
    # Reshape X to match the function's expected input format
    tau_values = np.array([tau_fn(np.array([x])) for x in X])
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(X, tau_values, 'lightblue', linewidth=2)
    plt.title('Treatment Effect Function')
    plt.xlabel('X[0] value')
    plt.ylabel('Treatment Effect')
    plt.grid(False)
    plt.show()


def plot_temporal_data(Y, T, X, treated_unit, competitor_idx, Y_c, num_unrelated=50, figsize=(8, 6)):

    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    time_steps = np.arange(Y.shape[1])
    
    # Plot competitors
    for idx in competitor_idx:
        ax.plot(time_steps, Y[idx], 'orange', alpha=0.4, label='Related Competitor', linewidth=1)
    
    # Plot unrelated units
    unrelated_mask = ~np.isin(np.arange(len(Y)), np.append(competitor_idx, treated_unit))
    unrelated_idx = np.random.choice(np.where(unrelated_mask)[0], num_unrelated)
    for idx in unrelated_idx:
        ax.plot(time_steps, Y[idx], 'lightblue', alpha=0.4, label='Unrelated', linewidth=1)
    
    # Plot treated unit and its counterfactual
    ax.plot(time_steps, Y[treated_unit], 'r-', label='Treated Unit', linewidth=1)
    ax.plot(time_steps, Y_c[treated_unit], 'r--', label='True Counterfactual', linewidth=1)
    
    
    # Add treatment start line
    ax.axvline(x=round(len(time_steps)//1.5), color='grey', linestyle='--', label='Treatment Start')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # Set labels and title
    ax.set_title('Outcomes Over Time')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Outcome Value')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_panel_data(Y, Y_c, T, X, treated_unit, competitor_idx, save_path=None):
    n_subjects, n_timesteps, n_features = X.shape
    
    # Create multi-index data
    panel_data = []
    
    for subject in range(n_subjects):
        # Determine unit type
        is_treated = (subject == treated_unit)
        is_competitor = subject in competitor_idx
        
        for time in range(n_timesteps):
            row = {
                'subject_id': subject,
                'time': time,
                'Y': Y[subject, time],
                'Y_c': Y_c[subject, time],
                'T': T[subject, time],
                'unit_type': 'treated' if is_treated else ('competitor' if is_competitor else 'other'),
                'is_post': time >= round(n_timesteps//1.5)
            }
            
            # Add features
            for j in range(n_features):
                row[f'X{j}'] = X[subject, time, j]
            
            panel_data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(panel_data)
    
    # # Set multi-index
    # df = df.set_index(['subject_id', 'time'])
    
    # Save if path provided
    if save_path:
        df.to_csv(save_path)
    
    return df
