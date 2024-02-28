import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly
import os


class WaveletDenoise:
    def __init__(self, df, metadata_columns, unique_id_col, class_label_col, save_path, wavelets, level=1, save_data=True, show_graph=True):
        """
        Initializes the WaveletDenoise class, setting up for denoising and visualization of the given dataset using wavelet transformations.

        Parameters:
        - df (pd.DataFrame): The dataset containing both metadata and data to be processed. Expected to have metadata in the first few columns followed by data.
        - metadata_columns (int): The number of columns at the beginning of the DataFrame that are treated as metadata and not to be denoised.
        - unique_id_col (str): The column name that uniquely identifies each sample in the dataset.
        - class_label_col (str): The column name that contains class labels for each sample.
        - save_path (str): The directory path where the denoised data and plots will be saved.
        - wavelets (list of str): A list of wavelet names as strings to be used for denoising.
        - level (int, optional): The level of decomposition for the wavelet transform. Defaults to 1.
        - save_data (bool, optional): Whether to save the denoised data to CSV files. Defaults to True.
        - show_graph (bool, optional): Whether to generate and save plots of the original vs denoised data. Defaults to True.

        Outputs:
        - Initializes a WaveletDenoise instance with specified parameters and preprocessed data.
        """
        self.df = df
        self.metadata_columns = metadata_columns
        self.unique_id_col = unique_id_col
        self.class_label_col = class_label_col
        self.save_path = save_path
        self.wavelets = wavelets
        self.level = level
        self.save_data = save_data
        self.show_graph = show_graph
        # Extract the data (excluding metadata) for processing and store metadata separately.
        self.data = df.iloc[:, metadata_columns:].to_numpy()
        self.subject_ids = df[unique_id_col].values
        self.vasoplegia_labels = df[class_label_col].values

    @staticmethod
    def madev(d, axis=None):
        """
        Calculate the mean absolute deviation of a dataset.

        Parameters:
        - d (np.array): The dataset for which the mean absolute deviation is calculated.
        - axis (int, optional): The axis along which the mean is computed. The default is to compute the mean of the flattened array.

        Returns:
        - The mean absolute deviation of the dataset.
        """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def wavelet_denoising(self, x, wavelet='db1'):
        """
        Performs wavelet denoising on a given signal using specified wavelet type.

        Parameters:
        - x (np.array): The signal to be denoised.
        - wavelet (str, optional): The type of wavelet to use for denoising. Defaults to 'db1'.

        Returns:
        - The denoised signal as a numpy array.
        """
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1 / 0.6745) * self.madev(coeff[-self.level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    def process_and_visualize(self):
        """
        Processes the dataset through wavelet denoising for each specified wavelet type and optionally saves the results and generates visualizations.

        This method iterates over each wavelet specified in the `wavelets` list, denoises the data using that wavelet, and then optionally saves the denoised data and generates a plot comparing the original and denoised signals.
        """
        for wav in self.wavelets:
            filtered_data = np.zeros_like(self.data)
            for i in range(self.data.shape[0]):
                filtered_data[i] = self.wavelet_denoising(self.data[i], wavelet=wav)[:-1]

            if self.save_data:
                self.save_denoised_data(filtered_data, wav)

            if self.show_graph:
                self.plot_data(filtered_data, wav)

    def save_denoised_data(self, filtered_data, wavelet_name):
        """
        Saves the denoised data to a CSV file, preserving the initial metadata columns.

        Parameters:
        - filtered_data (np.array): The denoised data to be saved.
        - wavelet_name (str): The name of the wavelet used for denoising, used in the filename.
        """
        denoised_df = pd.concat([self.df.iloc[:, :self.metadata_columns].reset_index(drop=True), pd.DataFrame(filtered_data, columns=self.df.columns[self.metadata_columns:])], axis=1)
        denoised_df.to_csv(os.path.join(self.save_path, f"cpmg_{wavelet_name}.csv"), index=False)

    def plot_data(self, filtered_data, wavelet_name):
        """
        Generates and saves a plot comparing the original and denoised data, including scaling, for each sample.

        Parameters:
        - filtered_data (np.array): The denoised data to be plotted.
        - wavelet_name (str): The name of the wavelet used for denoising, included in the plot title.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        scaled_filtered_data = scaler.fit_transform(filtered_data)

        fig = make_subplots(rows=2, cols=1)
        step = 4  # Adjust step to reduce clutter in plots

        for i in range(0, self.data.shape[0], step):
            vasoplegia_label = self.vasoplegia_labels[i]
            subject_id = self.subject_ids[i]
            # Add traces for original and filtered data in both scaled and unscaled forms.
            fig.add_trace(go.Scatter(x=list(range(self.data.shape[1])), y=self.data[i], mode='lines', name=f'Original {subject_id}', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(range(filtered_data.shape[1])), y=filtered_data[i], mode='lines', name=f'Filtered {subject_id}', line=dict(color='red')), row=1, col=1)
            # Repeat for scaled data
            fig.add_trace(go.Scatter(x=list(range(scaled_data.shape[1])), y=scaled_data[i], mode='lines', name=f'Scaled Original {subject_id}', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=list(range(scaled_filtered_data.shape[1])), y=scaled_filtered_data[i], mode='lines', name=f'Scaled Filtered {subject_id}', line=dict(color='purple')), row=2, col=1)

        # Update plot layout for clarity and aesthetics
        fig.update_xaxes(title_text='Features', row=1, col=1, matches='x')
        fig.update_yaxes(title_text='Signal Intensity', row=1, col=1)
        fig.update_xaxes(title_text='Features', row=2, col=1, matches='x')
        fig.update_yaxes(title_text='Scaled Signal Intensity', row=2, col=1)
        fig.update_layout(title_text=f'Comparison of Original and Denoised NMR Data for All Samples - Wavelet {wavelet_name}')
        plotly.offline.plot(fig, filename=os.path.join(self.save_path, f"cpmg_{wavelet_name}.html"), auto_open=False)
