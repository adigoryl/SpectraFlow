import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
import glob
import os
import re

from pathlib import Path
import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
import copy


def extract_values(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    values = content.split('Standard deviation values:')[1].strip().split('\n')
    mean_values = content.split('Standard deviation values:')[0].split('Mean values:')[1].strip().split('\n')

    mean_values_dict = dict((item.split()[0], float(item.split()[1])) for item in mean_values)
    std_dev_values_dict = dict((item.split()[0], float(item.split()[1])) for item in values)

    return mean_values_dict['val_f1_shallow'], mean_values_dict['val_auc_shallow'], std_dev_values_dict[
        'val_f1_shallow'], std_dev_values_dict['val_auc_shallow']


def sort_paths_by_epoch(paths):
    def extract_epoch_number(path):
        match = re.search(r'epoch(\d+)', path)
        return int(match.group(1)) if match else float('inf')

    return sorted(paths, key=extract_epoch_number)


def jaccard_similarity(list1: List[str], list2: List[str]) -> float:
    """
    Calculate Jaccard Similarity between two lists.

    The Jaccard similarity coefficient measures the size of the intersection divided by the size of the union of two sets.
    The resulting number is a scalar value representing the overall similarity (or stability) of the feature selection
    reflected by the different rankings. Note that the Jaccard similarity ranges from 0 to 1, where 1 signifies that
    the rankings are identical, and 0 signifies that the rankings do not share any features.

    Args:
        list1 (List[str]): First list of features
        list2 (List[str]): Second list of features

    Returns:
        float: Jaccard similarity between the two lists
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def average_jaccard_similarity(rankings: List[List[str]]) -> Tuple[float, float]:
    """
    Calculate the average Jaccard similarity and standard deviation for a list of rankings

    Args:
        rankings (List[List[str]]): List of rankings, where each ranking is a list of features

    Returns:
        Tuple[float, float]: Average Jaccard similarity and standard deviation across all pairs of rankings
    """
    jaccard_similarities = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            similarity = jaccard_similarity(rankings[i], rankings[j])
            jaccard_similarities.append(similarity)
    return round(np.mean(jaccard_similarities), 4), round(np.std(jaccard_similarities), 4)

# ----------------------------------------------------------
# PLOT THE SCORE AND SIMILARTY PER EPOCH, OR X EPOCH PER FEATURE SELECTED
# ----------------------
# a = "/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/Test"
# path_list = glob.glob(a + "/*/*/*/*/*/results_summary.txt")
# path_list = path_list[0:6]
# sorted_paths = sort_paths_by_epoch(path_list)
#
# result_lists = {
#     'mean_f1': [],
#     'mean_auc': [],
#     'std_f1': [],
#     'std_auc': [],
#     'avg_similarity': [],
#     'std_similarity': [],
#     'epoch_per_selected_feature': []
# }
#
# for path in sorted_paths:
#     path_components = path.split(os.sep)
#
#     mean_f1, mean_auc, std_f1, std_auc = extract_values(path)
#     result_lists['mean_f1'].append(mean_f1)
#     result_lists['mean_auc'].append(mean_auc)
#     result_lists['std_f1'].append(std_f1)
#     result_lists['std_auc'].append(std_auc)
#
#     # print("{}/{}/{}: mean f1 {} mean auc {} | std f1 {} std auc {}".format(path_components[8], path_components[10],
#     #                                                      path_components[11], a, b, c, d))
#
#     # Get the directory part of the old path
#     directory = os.path.dirname(path)
#
#     # Define the new file name
#     new_file_name = "selected_feature_indicies.backup"
#
#     # Join the directory with the new file name
#     rankings_path = os.path.join(directory, new_file_name)
#
#     with open(rankings_path, 'rb') as f:
#         rankings = pickle.load(f)
#
#     avg_similarity, std_dev = average_jaccard_similarity(rankings)
#     result_lists['avg_similarity'].append(avg_similarity)
#     result_lists['std_similarity'].append(std_dev)
#
#     epoch = int(re.findall(r'\d+', path_components[11])[0])
#     num_of_features_to_select = 50
#     result_lists['epoch_per_selected_feature'].append(epoch // num_of_features_to_select)
#
# # # Now you have your separate lists:
# # print(result_lists['mean_f1'])
# # print(result_lists['mean_auc'])
# # print(result_lists['std_f1'])
# # print(result_lists['std_auc'])
# # print(result_lists['avg_similarity'])
# # print(result_lists['std_similarity'])
# # print(result_lists['epoch_per_selected_feature'])
#
#
# def add_trace_with_std(fig, df, x, y, y_std, name, color, yaxis='y'):
#     lighter_color = [min(c + 50, 255) for c in color]
#     darker_color = [max(c - 50, 0) for c in color]
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y] + df[y_std], mode='lines',
#                              line=dict(color=f'rgba({lighter_color[0]}, {lighter_color[1]}, {lighter_color[2]}, 0.3)'),
#                              showlegend=False, yaxis=yaxis, name=name))
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y] - df[y_std], mode='lines',
#                              line=dict(color=f'rgba({lighter_color[0]}, {lighter_color[1]}, {lighter_color[2]}, 0.3)'),
#                              fill='tonexty', showlegend=False, yaxis=yaxis, name=name))
#
#     fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers',
#                              line=dict(color=f'rgb({darker_color[0]}, {darker_color[1]}, {darker_color[2]})'),
#                              name=name, yaxis=yaxis))
#
#
# df = pd.DataFrame(result_lists)
#
# fig = go.Figure()
#
# add_trace_with_std(fig, df, 'epoch_per_selected_feature', 'mean_auc', 'std_auc', 'Mean AUC', [31, 119, 180])
# add_trace_with_std(fig, df, 'epoch_per_selected_feature', 'avg_similarity', 'std_similarity', 'Avg Similarity', [255, 127, 14], 'y2')
#
# fig.update_layout(
#     xaxis=dict(title='Epoch per selected feature'),
#     yaxis=dict(title='Mean AUC'),
#     yaxis2=dict(
#         title='Avg Similarity',
#         overlaying='y',
#         side='right'
#     ),
#     template='plotly_white'
# )
#
# plotly.offline.plot(fig, filename='plot.html')


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_csv_file(file_path):
    return pd.read_csv(file_path)


def ranking_stats_to_txt(top_features, bins_to_names, save_path):
    """
    Process a DataFrame: add binned metabolites, binned metabolite range, sort by mean, and rename columns.

    Parameters:
    top_features (pandas.DataFrame): Input DataFrame.
    bins_to_names (dict): Dictionary for mapping bin numbers to feature names.

    Returns:
    pandas.DataFrame: The processed DataFrame.
    """
    a = top_features.reset_index()
    feature_names = []
    feature_range = []

    for top_fs_index in list(a["index"]):
        names = bins_to_names[top_fs_index]
        feature_names.append(names)
        feature_range.append("{}-{}".format(round(float(names[0]),4), round(float(names[-1]),4)))

    a['binned_metabolites'] = feature_names
    a['binned_metabolite_range'] = feature_range
    a = a.sort_values(by="mean")

    rename_dict = {
        "index": "bin_number",
        "mean": "mean_rank_position_across_samples",
        "frequency": "Frequency_count_across_sample_rankings"  # fixed a typo in your code
    }

    a = a.rename(columns=rename_dict)

    a.to_csv(save_path + '/selected_metabolite_ranges.csv', index=False)
    return a


def get_index_positions(data):
    index_positions = defaultdict(list)
    for i, sublist in enumerate(data):
        for j, item in enumerate(sublist):
            index_positions[item].append(j + 1)
    return index_positions


def create_stats_df(unique_features, index_positions):
    df = pd.DataFrame(index=unique_features, columns=['mean', 'max', 'min', 'Q1', 'Q3', 'frequency'])
    for feature in unique_features:
        positions = index_positions[feature]

        df.loc[feature, 'mean'] = np.mean(positions)
        df.loc[feature, 'max'] = np.max(positions)
        df.loc[feature, 'min'] = np.min(positions)
        df.loc[feature, 'Q1'] = np.percentile(positions, 25)
        df.loc[feature, 'Q3'] = np.percentile(positions, 75)
        df.loc[feature, 'frequency'] = len(positions)
    df['frequency'] = df['frequency'].astype(int)
    return df

# ----------------------------------------
# PLOT THE PREDICTION ACCURACY AND SIMILARITY FOR RECURSIVE FEATURE ELIMINATION AFTER SEQUENTIAL ATTENTION FEATURE SELECTION
# ---------------------------------
def plot_pred_acc_multimetric_RFE(df, ordered_ranking, save_path, auto_open, metric_dict={"rfe_val_auc_shallow": "AUC Accuracy"}):

    # Create stats_df for each metric
    stats_dfs = {metric: df.groupby('RFE_step')[metric].agg(['mean', 'std']) for metric in metric_dict.keys()}

    sins_list = []
    for i in list(range(len(ordered_ranking[0]), 0, -2)):
        a = [arr[0:i] for arr in ordered_ranking]
        avg_similarity, std_dev = average_jaccard_similarity(a)
        sins_list.append({
            "avg_similarity": avg_similarity,
            "std_dev": std_dev,
            "RFE_step": i
        })

    sims_df = pd.DataFrame(sins_list)

    def add_trace(fig, x, y, y_upper, y_lower, name, color, fillcolor, yaxis):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color), yaxis=yaxis))
        fig.add_trace(go.Scatter(x=x, y=y_upper, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis, showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis, showlegend=False))

    # Create a new figure
    fig = go.Figure()

    colors = [
        'rgba(0, 0, 255, 1)',   # Blue
        'rgba(0, 128, 0, 1)',  # Green
        'rgba(128, 0, 128, 1)',# Purple
        'rgba(255, 165, 0, 1)',# Orange
        'rgba(255, 0, 0, 1)'   # Red
    ]

    # Loop through each metric and add to the plot
    for idx, metric in enumerate(metric_dict.keys()):
        color = colors[idx % len(colors)]
        fillcolor = color.replace(", 1)", ", 0.2)")

        data_to_plot = {
            'x': stats_dfs[metric].index,
            'y': stats_dfs[metric]['mean'],
            'y_upper': stats_dfs[metric]['mean'] + stats_dfs[metric]['std'],
            'y_lower': stats_dfs[metric]['mean'] - stats_dfs[metric]['std'],
            'name': f"Mean {metric_dict[metric]}",
            'color': color,
            'fillcolor': fillcolor,
            'yaxis': 'y1'
        }

        add_trace(fig, **data_to_plot)

    # Adding ranking similarity
    add_trace(fig, sims_df['RFE_step'], sims_df['avg_similarity'], sims_df['avg_similarity'] + sims_df['std_dev'],
              sims_df['avg_similarity'] - sims_df['std_dev'], 'Mean Ranking Similarity', 'rgba(255, 0, 0, 1)', 'rgba(255, 0, 0, 0.2)',
              'y2')

    # Update layout to include a secondary Y-axis
    fig.update_layout(
        title='Prediction accuracy and ranking similarity across samples.',
        xaxis_title='Number of top Metabolic Features evaluated',
        yaxis_title='Prediction Accuracy',
        yaxis2=dict(title='Mean Ranking Similarity', overlaying='y', side='right'),
        width=1200,
        height=800,
    )

    file_name = "{}/acc_simi_multiple_metrics.html".format(save_path)
    plotly.offline.plot(fig, filename=file_name, auto_open=auto_open)


def plot_pred_acc_and_sim_for_RFE(df, ordered_ranking, save_path, auto_open, metric_dict={"rfe_val_auc_shallow": "AUC Accuracy"}):

    metric = list(metric_dict.keys())[0]
    stats_df = df.groupby('RFE_step')[metric].agg(['mean', 'std'])

    sins_list = []
    for i in list(range(len(ordered_ranking[0]), 0, -2)):
        a = [arr[0:i] for arr in ordered_ranking]
        avg_similarity, std_dev = average_jaccard_similarity(a)

        sins_list.append({
            "avg_similarity": avg_similarity,
            "std_dev": std_dev,
            "RFE_step": i
        })

    sims_df = pd.DataFrame(sins_list)


    def add_trace(fig, x, y, y_upper, y_lower, name, color, fillcolor, yaxis):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=color), yaxis=yaxis))
        fig.add_trace(go.Scatter(x=x, y=y_upper, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis,showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y_lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=fillcolor, yaxis=yaxis,showlegend=False))


    # Create a new figure
    fig = go.Figure()

    # List of data to add to the plot
    data_to_plot = [
        {
            'x': stats_df.index,
            'y': stats_df['mean'],
            'y_upper': stats_df['mean'] + stats_df['std'],
            'y_lower': stats_df['mean'] - stats_df['std'],
            'name': 'Mean Test Accuracy',
            'color': 'blue',
            'fillcolor': 'rgba(0, 0, 255, 0.2)',
            'yaxis': 'y1'
        },
        {
            'x': sims_df['RFE_step'],
            'y': sims_df['avg_similarity'],
            'y_upper': sims_df['avg_similarity'] + sims_df['std_dev'],
            'y_lower': sims_df['avg_similarity'] - sims_df['std_dev'],
            'name': 'Mean Ranking Similarity',
            'color': 'red',
            'fillcolor': 'rgba(255, 0, 0, 0.2)',
            'yaxis': 'y2'
        }
    ]

    # Add data to the plot
    for data in data_to_plot:
        add_trace(fig, **data)

    # Update layout to include a secondary Y-axis
    fig.update_layout(
        title='Prediction accuracy and ranking similarity across samples.',
        xaxis_title='Number of top Metabolic Features evaluated',
        yaxis_title=metric_dict[metric],
        yaxis2=dict(title='Mean Ranking Similarity', overlaying='y', side='right'),
        width=1200,
        height=800,
    )

    file_name = "{}/acc_simi__{}.html".format(save_path, metric)
    plotly.offline.plot(fig, filename=file_name, auto_open=auto_open)

# --------------------------------------------------------
# FEATURE RANKING PLOT
# ---------
def feature_ranking_plot(top_features, index_positions, bins_to_names, save_path, auto_open):

    top_features = top_features.sort_values(by='mean', ascending=False)

    # Create a horizontal box plot
    fig = go.Figure()

    for feature in top_features.index:
        # Append the frequency to the feature name
        ppm_range = bins_to_names[feature]
        if len(ppm_range) > 1:
            feature_name = "{}-{}ppm (frq{})".format(round(float(ppm_range[0]), 3), round(float(ppm_range[-1]), 3), top_features.loc[feature, 'frequency'])
        else:
            feature_name = "{} (frq{})".format(ppm_range[0], top_features.loc[feature, 'frequency'])

        fig.add_trace(go.Box(x=index_positions[feature], name=feature_name, orientation='h', showlegend=False, marker_color='blue'))

    fig.update_layout(
        xaxis_title="Ranking Position",
        yaxis_title="Metabolic Feature",
        # boxmode='group',  # group together boxes of the different traces for each value of x
        # autosize=False,
        width=800,
        height=1200,

    )

    file_name = "{}/feature_ranking.html".format(save_path)
    plotly.offline.plot(fig, filename=file_name, auto_open=auto_open)
    print("a")

# -------------------------------------
# PPM plot median and all samples
# --------------------------
def plot_ppm_with_selection(ds, pca_ds, top_features, bins_to_names, ppm_chart_title, save_path, auto_open, plot_median=False):
    def add_trace(fig, class_data, ppm_float, class_value, group, class_colour, row, col, showlegend=True, legendgroup=None):
        fig.add_trace(go.Scatter(x=ppm_float, y=class_data.values.flatten(), mode='lines', name=group, line=dict(color=class_colour), legendgroup=legendgroup, showlegend=showlegend), row=row, col=col)

    def add_vrect(fig, min_val, max_val, row, col):
        fig.add_vrect(x0=min_val, x1=max_val, fillcolor="green", opacity=0.25, line_width=1, row=row, col=col)

    vaso_class = {0.0: "green", 1.0: "red"}
    vaso_class1 = {0.0: "Vasoplegia Positive (plot 1)", 1.0: "Vasoplegia Negative (plot 1)"}
    vaso_class2 = {0.0: "Vasoplegia Positive (plot 2)", 1.0: "Vasoplegia Negative (plot 2)"}
    vaso_class3 = {0.0: "Vasoplegia Positive (plot 3)", 1.0: "Vasoplegia Negative (plot 3)"}

    scaler = StandardScaler()
    data = ds[ds.columns[2:]].to_numpy()
    data_scaled = scaler.fit_transform(data)  # Fit and transform the complete dataset
    ds_scaled = pd.DataFrame(data_scaled, index=ds.index, columns=ds.columns[2:])
    ds_scaled = pd.concat([ds[ds.columns[:2]], ds_scaled], axis=1)

    ppm_float = ds[ds.columns[2:]].columns.astype(float)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False)

    median_prefix = ""
    if plot_median:
        median_prefix = "Median "
        unique_classes = ds["Vasoplegia"].unique()
        for class_value in unique_classes:
            class_data = ds[ds["Vasoplegia"] == class_value][ds.columns[2:]]
            median_data = class_data.median()  # Compute median
            add_trace(fig, median_data, ppm_float, class_value, vaso_class1[class_value], vaso_class[class_value], 1, 1, showlegend=True, legendgroup=vaso_class1[class_value])  # No scaling for subplot 1

            class_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value][ds_scaled.columns[2:]]
            median_data_scaled = class_data_scaled.median()  # Compute median of scaled data
            add_trace(fig, median_data_scaled, ppm_float, class_value, vaso_class2[class_value], vaso_class[class_value], 2, 1, showlegend=True, legendgroup=vaso_class2[class_value])  # Scaling for subplot 2

            class_data = pca_ds[pca_ds["Vasoplegia"] == class_value][pca_ds.columns[2:]]
            median_pca_data = class_data.median()  # Compute median of PCA data
            add_trace(fig, median_pca_data, list(range(len(pca_ds.columns[2:]))), class_value, vaso_class3[class_value], vaso_class[class_value], 3, 1, showlegend=True, legendgroup=vaso_class3[class_value])  # No scaling for subplot 3
    else:
        unique_classes = ds["Vasoplegia"].unique()
        for class_value in unique_classes:
            group_data = ds[ds["Vasoplegia"] == class_value]
            group_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value]
            group_data_pca = pca_ds[pca_ds["Vasoplegia"] == class_value]

            for i, (row, row_scaled, row_pca) in enumerate(zip(group_data.iterrows(), group_data_scaled.iterrows(), group_data_pca.iterrows())):
                if i % 4 != 0:  # Skip rows that aren't every 4th row
                    continue

                show_legend = (i // 4) == 0

                add_trace(fig, pd.DataFrame(row[1][2:]).T, ppm_float, class_value, vaso_class1[class_value], vaso_class[class_value], 1, 1, showlegend=show_legend, legendgroup=vaso_class1[class_value])
                add_trace(fig, pd.DataFrame(row_scaled[1][2:]).T, ppm_float, class_value, vaso_class2[class_value], vaso_class[class_value], 2, 1, showlegend=show_legend, legendgroup=vaso_class2[class_value])
                add_trace(fig, pd.DataFrame(row_pca[1][2:]).T, list(range(len(pca_ds.columns[2:]))), class_value, vaso_class3[class_value], vaso_class[class_value], 3, 1, showlegend=show_legend,legendgroup=vaso_class3[class_value])

    top_features = top_features.sort_values(by='mean').reset_index()

    for top_fs_index in list(top_features["index"]):
        feature_names = bins_to_names[top_fs_index]
        feature_ppm_values = [float(feature) for feature in feature_names]
        ppm_min = min(feature_ppm_values)
        ppm_max = max(feature_ppm_values)
        add_vrect(fig, ppm_min, ppm_max, 1, 1)
        add_vrect(fig, ppm_min, ppm_max, 2, 1)
        add_vrect(fig, top_fs_index - 1, top_fs_index + 1, 3, 1)

    fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=1, col=1, matches='x')
    fig.update_yaxes(title_text=median_prefix + 'Signal Intensity', row=1, col=1)
    fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=2, col=1, matches='x')
    fig.update_yaxes(title_text=median_prefix + 'Scaled Signal Intensity', row=2, col=1)
    fig.update_xaxes(title_text=median_prefix + 'PCA Binned Features', row=3, col=1)
    fig.update_yaxes(title_text='PC1 Values', row=3, col=1)
    fig.update_layout(xaxis3=dict(range=[0, len(pca_ds.columns[2:]) - 1]), title=ppm_chart_title)

    file_name = "{}/{}_ppm_chart2.html".format(save_path, median_prefix)
    plotly.offline.plot(fig, filename=file_name, auto_open=auto_open)


def plot_ppm_with_selection_paper_modified(ds, pca_ds, top_features, bins_to_names, ppm_chart_title, save_path, auto_open, plot_median=False):
    def add_trace(fig, class_data, ppm_float, class_value, group, class_colour, row, col, showlegend=True, legendgroup=None):
        fig.add_trace(go.Scatter(x=ppm_float, y=class_data.values.flatten(), mode='lines', name=group, line=dict(color=class_colour), legendgroup=legendgroup, showlegend=showlegend), row=row, col=col)

    def add_vrect(fig, min_val, max_val, row, col):
        fig.add_vrect(x0=min_val, x1=max_val, fillcolor="green", opacity=0.25, line_width=1, row=row, col=col)

    vaso_class = {0.0: "green", 1.0: "red"}
    vaso_class1 = {0.0: "Vasoplegia Positive (plot 1)", 1.0: "Vasoplegia Negative (plot 1)"}
    vaso_class3 = {0.0: "Vasoplegia Positive (plot 2)", 1.0: "Vasoplegia Negative (plot 2)"}
    # vaso_class3 = {0.0: "Vasoplegia Positive (plot 3)", 1.0: "Vasoplegia Negative (plot 3)"}

    scaler = StandardScaler()
    data = ds[ds.columns[2:]].to_numpy()
    data_scaled = scaler.fit_transform(data)  # Fit and transform the complete dataset
    ds_scaled = pd.DataFrame(data_scaled, index=ds.index, columns=ds.columns[2:])
    ds_scaled = pd.concat([ds[ds.columns[:2]], ds_scaled], axis=1)

    ppm_float = ds[ds.columns[2:]].columns.astype(float)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False)

    median_prefix = ""
    if plot_median:
        median_prefix = "Median "
        unique_classes = ds["Vasoplegia"].unique()
        for class_value in unique_classes:
            class_data = ds[ds["Vasoplegia"] == class_value][ds.columns[2:]]
            median_data = class_data.median()  # Compute median
            add_trace(fig, median_data, ppm_float, class_value, vaso_class1[class_value], vaso_class[class_value], 1, 1, showlegend=True, legendgroup=vaso_class1[class_value])  # No scaling for subplot 1

            # class_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value][ds_scaled.columns[2:]]
            # median_data_scaled = class_data_scaled.median()  # Compute median of scaled data
            # add_trace(fig, median_data_scaled, ppm_float, class_value, vaso_class2[class_value], vaso_class[class_value], 2, 1, showlegend=True, legendgroup=vaso_class2[class_value])  # Scaling for subplot 2

            class_data = pca_ds[pca_ds["Vasoplegia"] == class_value][pca_ds.columns[2:]]
            median_pca_data = class_data.median()  # Compute median of PCA data
            add_trace(fig, median_pca_data, list(range(len(pca_ds.columns[2:]))), class_value, vaso_class3[class_value], vaso_class[class_value], 2, 1, showlegend=True, legendgroup=vaso_class3[class_value])  # No scaling for subplot 3
    else:
        unique_classes = ds["Vasoplegia"].unique()
        for class_value in unique_classes:
            group_data = ds[ds["Vasoplegia"] == class_value]
            group_data_scaled = ds_scaled[ds_scaled["Vasoplegia"] == class_value]
            group_data_pca = pca_ds[pca_ds["Vasoplegia"] == class_value]

            for i, (row, row_scaled, row_pca) in enumerate(zip(group_data.iterrows(), group_data_scaled.iterrows(), group_data_pca.iterrows())):
                if i % 1 != 0:  # Skip rows that aren't every 4th row
                    continue

                show_legend = (i // 1) == 0

                add_trace(fig, pd.DataFrame(row[1][2:]).T, ppm_float, class_value, vaso_class1[class_value], vaso_class[class_value], 1, 1, showlegend=show_legend, legendgroup=vaso_class1[class_value])
                # add_trace(fig, pd.DataFrame(row_scaled[1][2:]).T, ppm_float, class_value, vaso_class2[class_value], vaso_class[class_value], 2, 1, showlegend=show_legend, legendgroup=vaso_class2[class_value])
                add_trace(fig, pd.DataFrame(row_pca[1][2:]).T, list(range(len(pca_ds.columns[2:]))), class_value, vaso_class3[class_value], vaso_class[class_value], 2, 1, showlegend=show_legend,legendgroup=vaso_class3[class_value])

    top_features = top_features.sort_values(by='mean').reset_index()

    for top_fs_index in list(top_features["index"]):
        feature_names = bins_to_names[top_fs_index]
        feature_ppm_values = [float(feature) for feature in feature_names]
        ppm_min = min(feature_ppm_values)
        ppm_max = max(feature_ppm_values)
        add_vrect(fig, ppm_min, ppm_max, 1, 1)
        # add_vrect(fig, ppm_min, ppm_max, 2, 1)
        add_vrect(fig, top_fs_index - 1, top_fs_index + 1, 2, 1)

    fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=1, col=1, matches='x')
    fig.update_yaxes(title_text=median_prefix + 'Signal Intensity', row=1, col=1)
    # fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=2, col=1, matches='x')
    # fig.update_yaxes(title_text=median_prefix + 'Scaled Signal Intensity', row=2, col=1)
    fig.update_xaxes(title_text=median_prefix + 'PCA Binned Features', row=2, col=1)
    fig.update_yaxes(title_text='PC1 Values', row=2, col=1)
    fig.update_layout(xaxis3=dict(range=[0, len(pca_ds.columns[2:]) - 1]), title=ppm_chart_title)

    file_name = "{}/{}_ppm_chart2.html".format(save_path, median_prefix)
    plotly.offline.plot(fig, filename=file_name, auto_open=auto_open)


def generate_plots(datasets_root, denoised_dataset_name, computations_root, dataset_name, auto_open=False):

    save_path = computations_root + "/plots"
    os.makedirs(save_path, exist_ok=True)

    # Dataset with the original ppm
    original_data_df = load_csv_file('{}/{}.csv'.format(datasets_root, denoised_dataset_name))

    # Dataset with transformed ppm ranges (PCA bins)
    pca_binned_data_df = load_csv_file('{}/{}.csv'.format(datasets_root, dataset_name))

    # Ordered rankings from samples
    ordered_ranking = load_pickle_file("{}/features_in_selection_order.backup".format(computations_root))

    # For some reason, the sequental attention selects the same feature at the beginning of the training
    #  (perhaps it's not trained enough when performing first feature selection) -> drop this feature
    ordered_ranking = [sublist[1:] for sublist in ordered_ranking]

    # Bin indexes to actual ppm ranges
    if "peak" in dataset_name:
        bins_to_names = load_pickle_file('{}/{}.peak_to_names'.format(datasets_root, dataset_name))
    elif "bin" in dataset_name:
        bins_to_names = load_pickle_file('{}/{}.bins_to_names'.format(datasets_root, dataset_name))
    else:
        column_names = original_data_df.columns[2:]
        bins_to_names = {i: [name] for i, name in enumerate(column_names)}


    # Predictions from sample RFE
    rfe_performance_df = load_pickle_file("{}/rfe_performance_scores.backup".format(computations_root))

    # Predictions from Leave-one-out RFE (merged rankings based on samples not used for training)
    loo_rfe_eval_df = load_pickle_file("{}/loo_rfe_eval.backup".format(computations_root))

    # ------------------------------------
    data = list(ordered_ranking)
    unique_features = list(set(item for sublist in data for item in sublist))
    # Since the rankings are ordered, we can calculate the respective feature positiions across all sample rankings
    index_positions = get_index_positions(data)
    df = create_stats_df(unique_features, index_positions)
    # Multiple samples will cause the ranking exceed the selection threshold, thus we shrink based on the frequency cocurrance across all sample rankings
    top_features = df.nlargest(len(ordered_ranking[0]), 'frequency')

    # Save the metabolite selection
    # ranking_stats_to_txt(top_features, bins_to_names, save_path)

    # # Plots the selected features on original dataset, scaled and PCA binned.
    # plot_ppm_with_selection(original_data_df, pca_binned_data_df, top_features, bins_to_names, "", save_path=save_path, auto_open=auto_open, plot_median=False)
    plot_ppm_with_selection_paper_modified(original_data_df, pca_binned_data_df, top_features, bins_to_names, "", save_path=save_path, auto_open=auto_open, plot_median=False)
    #
    # Plots the feature selection statistics across all samples (interquartile ranges)
    # feature_ranking_plot(top_features, index_positions, bins_to_names, save_path=save_path, auto_open=auto_open)

    # # AUC
    # plot_pred_acc_and_sim_for_RFE(df=rfe_performance_df, ordered_ranking=ordered_ranking, save_path=save_path, auto_open=auto_open, metric_dict={"rfe_val_auc_shallow": "AUC Accuracy"})
    #
    # # F1
    # plot_pred_acc_and_sim_for_RFE(df=rfe_performance_df, ordered_ranking=ordered_ranking, save_path=save_path, auto_open=auto_open, metric_dict={"rfe_val_f1_shallow": "F1 Accuracy"})
    #
    # plot_pred_acc_multimetric_RFE(
    #     df=rfe_performance_df,
    #     ordered_ranking=ordered_ranking,
    #     save_path=save_path,
    #     auto_open=auto_open,
    #     metric_dict={
    #         # 'rfe_val_f1_shallow': 'F1',
    #         # 'rfe_val_loss_shallow': 'Loss',
    #         # 'rfe_val_auc_shallow': 'ROC AUC',
    #         'rfe_val_positive_precision_shallow': 'Pos Precision',
    #         'rfe_val_positive_recall_shallow': 'Pos Recall',
    #         'rfe_val_negative_precision_shallow': 'Neg Precision',
    #         'rfe_val_negative_recall_shallow': 'Neg Recall'
    #     }
    # )
    #
    # #
    # # # AUC LOO
    # plot_pred_acc_and_sim_for_RFE(df=loo_rfe_eval_df, ordered_ranking=ordered_ranking, save_path=save_path, auto_open=auto_open, metric_dict={"roc_auc": "AUC Accuracy"})
    # #
    # # # F1 LOO
    # plot_pred_acc_and_sim_for_RFE(df=loo_rfe_eval_df, ordered_ranking=ordered_ranking, save_path=save_path, auto_open=auto_open, metric_dict={"f1": "F1 Accuracy"})
    #
    # plot_pred_acc_multimetric_RFE(
    #     df=loo_rfe_eval_df,
    #     ordered_ranking=ordered_ranking,
    #     save_path=save_path,
    #     auto_open=auto_open,
    #     metric_dict={
    #         # 'f1': 'F1',
    #         # 'roc_auc': 'ROC AUC',
    #         'positive_precision': 'Pos Precision',
    #         'positive_recall': 'Pos Recall',
    #         'negative_precision': 'Neg Precision',
    #         'negative_recall': 'Neg Recall'
    #     }
    # )

    return None


# generate_plots(
#     datasets_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/datasets/vasoplegia/different_wavelets",
#     denoised_dataset_name="cpmg_db4",
#     computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_no_pre_wavelets_test/10_FOLDS_5_REPS/2023-07-26_16:12/cpmg_db4__pca_reduction__bin005_overlap0025/epoch500/no_scaler",
#     dataset_name="cpmg_db4__pca_reduction__bin005_overlap0025",
#     auto_open=True
# )

# # ----------------------------------
# # Example for targeted dataset (commet out one the ppm plot and ranking_stats_to_txt)
# generate_plots(
#     datasets_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/datasets/vasoplegia/",
#     denoised_dataset_name="CPMG_targeted_clean",
#     computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_no_pre_wavelets_test/10_FOLDS_5_REPS/2023-08-04_16:39/CPMG_targeted_clean/epoch100/auto_scaler",
#     dataset_name="CPMG_targeted_clean",
#     auto_open=True
# )

#

# -------------------------------------------------- Untargeted -----------------------------------------
# Applies above to multiple dataset results
dataset_names = [
    # "cpmg_db1__pca_reduction__bin005_overlap0025",
    # "cpmg_db2__pca_reduction__bin005_overlap0025",
    # "cpmg_db4__pca_reduction__bin005_overlap0025",
    # "cpmg_db7__pca_reduction__bin005_overlap0025",
    "cpmg_db20__pca_reduction__bin005_overlap0025",
    # "cpmg_db38__pca_reduction__bin005_overlap0025",
]

extracted_names = [name.split('__')[0] for name in dataset_names]

for i, dn in enumerate(dataset_names):
    denoised_binned_name = "{}__pca_reduction__bin005_overlap0025".format(extracted_names[i])
    generate_plots(
        datasets_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/datasets/vasoplegia/different_wavelets",
        denoised_dataset_name=extracted_names[i],
        # computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_no_pre_wavelets_test/10_FOLDS_5_REPS/2023-07-27_12:00/{}/epoch500/no_scaler".format(denoised_binned_name),
        computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_CPMG_untargeted_db_test/10_FOLDS_10_REPS/2023-08-11_17:36/{}/epoch800/auto_scaler".format(denoised_binned_name),
        # computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_CPMG_untargeted_db_test/10_FOLDS_10_REPS/2023-08-14_16:18/{}/epoch800/auto_scaler".format(denoised_binned_name),
        # computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/More_metrics/CPMG_untargeted/10_FOLDS_10_REPS/2023-08-18_18:07/{}/epoch800/auto_scaler".format(denoised_binned_name),
        dataset_name=denoised_binned_name,
        auto_open=True
    )


# # Version without binning
# # Applies above to multiple dataset results
# dataset_names = [
#     "cpmg_db1__pca_reduction__bin005_overlap0025",
#     "cpmg_db4__pca_reduction__bin005_overlap0025",
#     "cpmg_db7__pca_reduction__bin005_overlap0025",
#     "cpmg_db20__pca_reduction__bin005_overlap0025",
#     "cpmg_db38__pca_reduction__bin005_overlap0025",
# ]
#
# extracted_names = [name.split('__')[0] for name in dataset_names]
#
# for i, dn in enumerate(extracted_names):
#     # denoised_binned_name = "{}__pca_reduction__bin005_overlap0025".format(extracted_names[i])
#     generate_plots(
#         datasets_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/datasets/vasoplegia/different_wavelets",
#         denoised_dataset_name=dn,
#         # computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_no_pre_wavelets_test/10_FOLDS_5_REPS/2023-07-27_12:00/{}/epoch500/no_scaler".format(denoised_binned_name),
#         computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_CPMG_untargeted_db_test/10_FOLDS_10_REPS/2023-08-11_17:36/{}/epoch800/auto_scaler".format(dn),
#         dataset_name=dn,
#         auto_open=True
#     )

# --------------------------------
# Version for targeted dataset
# -------------------------------

# epoch = [
#     # "100", "200", "300", "400", "500",
#     "800"
# ]
#
#
# for i, e in enumerate(epoch):
#     generate_plots(
#         datasets_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/datasets/vasoplegia",
#         denoised_dataset_name="CPMG_targeted_clean",
#         # computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/LOO_RFE_seq_attention_CPMG_untargeted_db_test/10_FOLDS_10_REPS/2023-08-11_17:36/{}/epoch800/auto_scaler".format(dn),
#         computations_root="/Users/aw678/PycharmProjects/sequential_attention/sequential_attention/experiments/More_metrics/CPMG_targeted_clean/10_FOLDS_10_REPS/2023-08-18_17:21/CPMG_targeted_clean/epoch{}/auto_scaler".format(e),
#         dataset_name="CPMG_targeted_clean",
#         auto_open=True
#     )
