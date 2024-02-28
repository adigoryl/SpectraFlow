import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import plotly.io as pio


def plot_ppm_with_selection(dataset_path, selected_feature_indicies, ppm_chart_title, SAVE_PATH, plot_median=False, bin=0.005, display_top_n_bin=20):
    vaso_class = {0.0: "green", 1.0: "red"}
    vaso_class2 = {0.0: "Pos", 1.0: "Neg"}

    ds = pd.read_csv(dataset_path)

    data = ds[ds.columns[2:]].to_numpy()
    n_samples, n_features = data.shape

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    ppm = ds[ds.columns[2:]].columns
    ppm_float = ppm.astype(float)

    fig = make_subplots(rows=2, cols=1)

    if plot_median:
        unique_classes = ds["Vasoplegia"].unique()
        for class_value in unique_classes:
            class_data = ds[ds["Vasoplegia"] == class_value][ds.columns[2:]]
            class_data_scaled = scaler.transform(class_data)
            class_median = class_data.median()
            class_median_scaled = np.median(class_data_scaled, axis=0)
            class_colour = vaso_class[class_value]
            group = vaso_class2[class_value]

            fig.add_trace(go.Scatter(x=ppm_float, y=class_median, mode='lines', name=group,
                                     line=dict(color=class_colour), legendgroup=group), row=1, col=1)
            fig.add_trace(go.Scatter(x=ppm_float, y=class_median_scaled, mode='lines', name="Scaled " + group,
                                     line=dict(color=class_colour), legendgroup="Scaled " + group), row=2, col=1)
    else:
        for i in range(n_samples):
            class_colour = vaso_class[ds.iloc[i]["Vasoplegia"]]
            group = vaso_class2[ds.iloc[i]["Vasoplegia"]]
            fig.add_trace(go.Scatter(x=ppm_float, y=data[i], mode='lines', name=group, line=dict(color=class_colour),
                                     legendgroup=group), row=1, col=1)
            fig.add_trace(go.Scatter(x=ppm_float, y=scaled_data[i], mode='lines', name="Scaled " + group,
                                     line=dict(color=class_colour), legendgroup="Scaled " + group), row=2, col=1)

    fig.update_xaxes(title_text='Features', row=1, col=1)
    fig.update_yaxes(title_text='Signal Intensity', row=1, col=1)
    fig.update_xaxes(title_text='Features', row=2, col=1)
    fig.update_yaxes(title_text='Scaled Signal Intensity', row=2, col=1)
    fig.update_layout(title=ppm_chart_title)

    # Your NMR dataset
    ppm_data = ppm_float  # Replace with your ppm data

    # Your list of lists of feature indices
    feature_indices_lists = selected_feature_indicies  # Replace with your list of lists of feature indices

    # Create bins from 0 to 10 with step size 0.005
    # bins = np.arange(0, 10.005, 0.005)

    bins = np.arange(10, -(1 + bin), -bin)

    # Flatten the list of lists of feature indices
    flattened_indices = [index for sublist in feature_indices_lists for index in sublist]

    # Bin the data
    bin_counts = np.zeros(len(bins) - 1)
    for index in flattened_indices:
        ppm_value = ppm_data[index]
        bin_index = np.digitize(ppm_value, bins) - 1
        bin_counts[bin_index] += 1

    # Identify N bins with the highest count

    bin_counts_df = pd.DataFrame({"Bin_Start": bins[:-1], "Bin_End": bins[1:], "Count": bin_counts})
    top_N_bins = bin_counts_df.nlargest(display_top_n_bin, 'Count')
    print("Top N Bins:")
    print(top_N_bins)

    # Identify the starting and ending indices on the original ppm for all N bins with the highest count
    ppm_index_ranges = []
    for _, row in top_N_bins.iterrows():
        start_ppm, end_ppm, count = row['Bin_Start'], row['Bin_End'], row["Count"]
        # The ppm is in descending order, so the "end_ppm" smaller than "start_ppm"
        start_indices = np.where((ppm_data >= end_ppm) & (ppm_data < start_ppm))[0]
        if len(start_indices) > 0:
            ppm_index_ranges.append((start_indices[0], start_indices[-1], count))

    print("Starting and ending indices for the top N bins:")
    for i, (start_idx, end_idx, count) in enumerate(ppm_index_ranges):
        ppm_left = float(ppm[start_idx])
        ppm_right = float(ppm[end_idx])

        fig.add_vrect(x0=ppm_left, x1=ppm_right,
                      annotation_text="{}".format(int(count)),
                      fillcolor="green", opacity=0.25, line_width=1)

    fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=1, col=1)
    fig.update_xaxes(autorange="reversed", dtick=0.25, title_text='PPM Range', row=2, col=1)

    fig.write_html("{}/ppm_plot.html".format(SAVE_PATH))
    fig.write_image("{}/ppm_plot.png".format(SAVE_PATH))

    return top_N_bins





#
# def plot_ppm_with_selection(dataset_path, selected_feature_indicies, ppm_chart_title, SAVE_PATH):
#
#     vaso_class = {0.0: "green", 1.0: "red"}
#     vaso_class2 = {0.0: "Pos", 1.0: "Neg"}
#     dataset_name = "cpmg_denoised_normalised"
#
#     ds = pd.read_csv(dataset_path)
#
#
#     data = ds[ds.columns[2:]].to_numpy()
#     n_samples, n_features = data.shape
#
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
#
#     # Get ppm names for x-axis
#     ppm = ds[ds.columns[2:]].columns
#     ppm_float = ppm.astype(float)
#
#     # Create two subplots
#     fig = make_subplots(rows=2, cols=1)
#
#     # Plot the original data and filtered data in the first subplot
#     for i in range(n_samples):
#         class_colour = vaso_class[ds.iloc[i]["Vasoplegia"]]
#         group = vaso_class2[ds.iloc[i]["Vasoplegia"]]
#         fig.add_trace(go.Scatter(x=ppm_float, y=data[i], mode='lines', name=group, line=dict(color=class_colour),
#                                  legendgroup=group), row=1, col=1)
#         fig.add_trace(go.Scatter(x=ppm_float, y=scaled_data[i], mode='lines', name="Scaled " + group,
#                                  line=dict(color=class_colour), legendgroup="Scaled " + group), row=2, col=1)
#
#     # Update the layout
#     fig.update_xaxes(title_text='Features', row=1, col=1)
#     fig.update_yaxes(title_text='Signal Intensity', row=1, col=1)
#     fig.update_xaxes(title_text='Features', row=2, col=1)
#     fig.update_yaxes(title_text='Scaled Signal Intensity', row=2, col=1)
#     fig.update_layout(title=ppm_chart_title)
#
#     colours = px.colors.qualitative.Plotly
#     for i, sample_fs in enumerate(selected_feature_indicies):
#         for feature_inx in sample_fs:
#             ppm_left = float(ppm[feature_inx - 1])
#             ppm_right = float(ppm[feature_inx + 1])
#
#             fig.add_vrect(x0=ppm_left, x1=ppm_right,
#                           annotation_text="{}".format(i + 1),
#                           fillcolor=colours[i], opacity=0.25, line_width=1)
#
#     fig.write_html("{}/ppm_plot.html".format(SAVE_PATH))
#     fig.write_image("{}/ppm_plot.png".format(SAVE_PATH))
