import pandas as pd
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns
style.use('ggplot')
color_list = sns.color_palette('Set2', n_colors=9)

from src.utils.dgp import get_market_data_generator, plot_tau_function, plot_temporal_data,create_panel_data
#import models
### SC
from src.models.sc import *
### AR(1)
from src.models.linear import *
### ARIMA

### LightGBM

### TFT

### TimeGPT


### Evaluation ---

# def plot_results(results_df, comparison='Increase Trend'):
#     """Plot the metrics for all setups and comparisons, averaging across setups."""
#     metrics = ['mse_sc_avg', 'mse_ar1_avg']  # Focusing on MSE for simplicity
#     method_labels = ['Synthetic Control', 'AR(1)']

#     if comparison == 'Increase Trend':
#         plt.figure(figsize=(12, 3))
#         for metric, label in zip(metrics, method_labels):
#             subset = results_df.groupby('increase_trend').mean(numeric_only=True).reset_index()
#             plt.plot(
#                 subset['increase_trend'], 
#                 subset[metric], 
#                 label=label
#             )
#         plt.xlabel("Increase Trend")
#         plt.ylabel("MSE")
#         plt.title("MSE Across Increase Trends (Averaged Over Setups)")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     elif comparison == 'Sigma':
#         plt.figure(figsize=(12, 3))
#         for metric, label in zip(metrics, method_labels):
#             subset = results_df.groupby('sigma').mean(numeric_only=True).reset_index()
#             plt.plot(
#                 subset['sigma'], 
#                 subset[metric], 
#                 label=label
#             )
#         plt.xlabel("Sigma")
#         plt.ylabel("MSE")
#         plt.title("MSE Across Sigma (Averaged Over Setups)")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     elif comparison == 'Number of Competitors':
#         plt.figure(figsize=(12, 3))
#         for metric, label in zip(metrics, method_labels):
#             subset = results_df.groupby('n_competitors').mean(numeric_only=True).reset_index()
#             plt.plot(
#                 subset['n_competitors'], 
#                 subset[metric], 
#                 label=label
#             )
#         plt.xlabel("Number of Competitors")
#         plt.ylabel("MSE")
#         plt.title("MSE Across Number of Competitors (Averaged Over Setups)")
#         plt.legend()
#         plt.grid()
#         plt.show()

#     elif comparison == 'Divergence':
#         plt.figure(figsize=(12, 3))
#         for metric, label in zip(metrics, method_labels):
#             subset = results_df.groupby('divergence').mean(numeric_only=True).reset_index()
#             plt.plot(subset['divergence'], subset[metric], label=label)
#         plt.xlabel("Divergence")
#         plt.ylabel("MSE")
#         plt.title("MSE Across Divergence (Averaged Over Setups)")
#         plt.legend()
#         plt.grid()
#         plt.show()

def calculate_true_effects(df_post, treated_unit, tau_fn):
    """Calculate true treatment effects"""
    true_effects = []
    post_periods = sorted(df_post['time'].unique())
    for t in post_periods:
        X_t = df_post[df_post['time'] == t].loc[
            df_post['subject_id'] == treated_unit, 
            ['X0', 'X1', 'X2', 'X3', 'X4']
        ].values[0]
        true_effects.append(tau_fn(X_t))
    return np.array(true_effects)

def calculate_metrics(estimated_effects, true_effects):
    """Calculate performance metrics"""
    mse = np.mean((estimated_effects - true_effects) ** 2)
    mae = np.mean(np.abs(estimated_effects - true_effects))
    bias = np.mean(estimated_effects - true_effects)
    return {'mse': mse, 'mae': mae, 'bias': bias}


def plot_predictions(treated, post_periods, treated_counterfactual, predictions_dict,color_list = color_list):
    """Plot predictions from different methods
    """
    plt.figure(figsize=(8, 6))
    
    # Plot full observed trajectory
    plt.plot(treated['time'], treated['Y'], label='Treated')
    
    # Plot post-treatment predictions and counterfactual
    for i, (method_name, predictions) in enumerate(predictions_dict.items()):
        plt.plot(post_periods, predictions, label=f'Predicted ({method_name})', 
        color=color_list[i], linewidth=2, linestyle='--')
    
    # Plot true counterfactual
    plt.plot(post_periods, treated_counterfactual, 
            label='True Counterfactual',
            color='red', linestyle=':')
    
    # Add treatment start line
    treatment_start = min(post_periods)
    plt.axvline(x=treatment_start, color='grey', 
                linestyle='--', label='Treatment Start')
    
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title('Treated Unit vs Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_effects(post_periods, true_effects, effects_dict, color_list = color_list):
    """Plot treatment effects from different methods"""
    plt.figure(figsize=(8, 6))
    time_points = np.arange(len(post_periods))
    
    for i, (method_name, estimated_effects) in enumerate(effects_dict.items()):
        plt.plot(time_points, estimated_effects, label=f'Estimated Effect ({method_name})', 
        color=color_list[i], linewidth=2,linestyle='--')
    
    plt.plot(time_points, true_effects, 'orange', label='True Effect', linewidth=2)
    plt.title('Estimated vs True Treatment Effects')
    plt.xlabel('Time since treatment')
    plt.ylabel('Effect Size')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()

def evaluate_methods(df, treated_unit, competitor_idx, tau_fn, methods=['SC', 'AR1'], show=True):
    """Main evaluation function supporting multiple methods"""
    # Prepare data
    treated = df[df['subject_id'] == treated_unit]
    df_pre = df[df['is_post'] == False]
    df_post = df[df['is_post'] == True]
    post_periods = sorted(df_post['time'].unique())
    
    # Get outcomes for post-treatment period only
    df_post_Y = df_post.pivot(index='time', columns='subject_id')['Y']
    treated_post_outcomes = df_post_Y[treated_unit].values
    treated_counterfactual = df_post.pivot(index='time', columns='subject_id')['Y_c'][treated_unit].values
    
    # Calculate true effects
    true_effects = calculate_true_effects(df_post, treated_unit, tau_fn)
    
    # Initialize results
    predictions_dict = {}
    effects_dict = {}
    metrics_dict = {}
    
    # Fit models and calculate effects
    for method in methods:
        if method == 'SC':
            predictions = fit_synthetic_control(df, treated_unit)
            predictions_dict['SC'] = predictions
            effects_dict['SC'] = treated_post_outcomes - predictions
        
        elif method == 'AR1':
            predictions = fit_ar1(df, treated_unit)
            predictions_dict['AR1'] = predictions
            effects_dict['AR1'] = treated_post_outcomes - predictions
    
    # Calculate metrics for each method
    for method in methods:
        metrics_dict[method] = calculate_metrics(effects_dict[method], true_effects)
    
    # Plotting
    if show:
        plot_predictions(treated, post_periods, treated_counterfactual, predictions_dict)
        plot_effects(post_periods, true_effects, effects_dict)
        
        # Print metrics
        for method in methods:
            print(f"\nPerformance Metrics ({method}):")
            metrics = metrics_dict[method]
            print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, Bias: {metrics['bias']:.4f}")
    
    return {
        'metrics': metrics_dict,
        'estimated_effects': effects_dict,
        'true_effects': true_effects,
        'predictions': predictions_dict
    }

def run_simulation(n_subjects, n_timesteps, d, sigmas, n_competitors_list, 
                  competitor_correlation, setups, increase_trends, divergence_trends, 
                  methods=['SC', 'AR1'], n_runs=10, save_df=False):
    """Run simulation with multiple parameters and methods
    
    Args:
        n_subjects (int): Number of total subjects
        n_timesteps (int): Number of time steps
        d (int): Number of features
        sigmas (list): List of noise levels to test
        n_competitors_list (list): List of number of competitors to test
        competitor_correlation (float): Correlation between competitors
        setups (list): List of setup types to test
        increase_trends (list): List of trend increases to test
        divergence_trends (list): List of divergence trends to test
        methods (list): List of methods to evaluate
        n_runs (int): Number of simulation runs per configuration
        save_df (bool): Whether to save generated data
        
    Returns:
        pd.DataFrame: Results of all simulations
    """
    results = []  # Store results for all simulations

    for sigma in sigmas:
        for n_competitors in n_competitors_list:
            for setup in setups:
                for trend in increase_trends:
                    for divergence in divergence_trends:
                        # Initialize metric storage for each method
                        metrics_lists = {
                            method: {
                                'mse': [], 'mae': [], 'bias': []
                            } for method in methods
                        }

                        for run in range(n_runs):
                            # Generate data
                            gen_data_fn, base_fn, tau_fn = get_market_data_generator(
                                n_subjects, n_timesteps, d, sigma, 
                                n_competitors=n_competitors, 
                                competitor_correlation=competitor_correlation, 
                                setup=setup, increase_trend=trend, 
                                divergence=divergence
                            )
                            Y, T, X, treated_unit, competitor_idx, Y_c = gen_data_fn()

                            # Create panel data
                            df = create_panel_data(Y, Y_c, T, X, treated_unit, competitor_idx)

                            if save_df:
                                df.to_csv(
                                    f'../data/df_sigma{sigma}_n{n_competitors}_setup{setup}_'
                                    f'trend{trend}_divergence{divergence}_n{run}.csv', 
                                    index=False
                                )

                            # Evaluate methods
                            results_dict = evaluate_methods(
                                df, treated_unit, competitor_idx, tau_fn, 
                                methods=methods, show=False
                            )

                            # Store metrics for each method
                            for method in methods:
                                method_metrics = results_dict['metrics'][method]
                                metrics_lists[method]['mse'].append(method_metrics['mse'])
                                metrics_lists[method]['mae'].append(method_metrics['mae'])
                                metrics_lists[method]['bias'].append(method_metrics['bias'])

                        # Calculate average results for this configuration
                        result_dict = {
                            'setup': setup,
                            'sigma': sigma,
                            'n_competitors': n_competitors,
                            'increase_trend': trend,
                            'divergence': divergence,
                        }

                        # Add metrics for each method
                        for method in methods:
                            result_dict.update({
                                f'mse_{method}_avg': np.mean(metrics_lists[method]['mse']),
                                f'mse_{method}_std': np.std(metrics_lists[method]['mse']),
                                f'mae_{method}_avg': np.mean(metrics_lists[method]['mae']),
                                f'bias_{method}_avg': np.mean(metrics_lists[method]['bias'])
                            })

                        results.append(result_dict)

    return pd.DataFrame(results)