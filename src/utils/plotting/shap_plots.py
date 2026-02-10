import matplotlib.pyplot as plt
import numpy as np
import shap


def shap_summary_plot_feature(
    shapley_values: list,
    features_arrays: list,
    feature_names: list,
    time_point: int,
    title: str,
    time_label: str,
    path_plots: str
    ):
    """
    Produce and save a SHAP summary_plot (aka beeswarm plot).
    Args:
        - shapley_values: list of arrays with shapley values
        - features_arrays: list of arrays with features
        - feature_names: list of arrays with feature names
        - time_point: time to plot
        - title: name of feature set to include in the plot title
    
    Example:
        summary_plot_feature(
            [shapley_values_per_feature[0]],
            [dict_arrays["genes"]],
            [data.features_names["genes_names"]],
            time_point=0,
            title="Genes"
        )
    """
    aggregated_values = []
    aggregated_features = []
    for index, shapley_value in enumerate(shapley_values):
        n_features = features_arrays[index].shape[-1]
        aggregated_values.append(shapley_value.mean(axis=0)[..., time_point].reshape(-1, n_features))
        aggregated_features.append(features_arrays[index].reshape(-1, n_features))
    agg_shapley_values = np.concatenate(aggregated_values, axis=-1)
    agg_features_array = np.concatenate(aggregated_features, axis=-1)
    feature_names = np.concatenate(feature_names, axis=-1)

    fig = plt.figure()
    shap.summary_plot(
        # feature, folds, last is time. Here averaging over folds
        agg_shapley_values,
        features=agg_features_array,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"{title} shapley values - Time {time_label}", fontsize=10)
    fig.savefig(f"{path_plots}/{title}_shap_time_{time_point}.pdf", format="pdf")
    plt.close()
