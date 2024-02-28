import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

class PCABinning:
    def __init__(self, file_path, root, new_dataset_name_prefix, metadata_columns=2, min_features=8, step_size=0.005, overlap=0.0025, n_components=2):
        """
        Initializes the PCABinning class for dimensionality reduction of spectroscopic data using PCA on defined bins.

        Parameters:
        - file_path (str): Path to the dataset file (CSV) to be processed.
        - root (str): Root directory where the output files will be saved.
        - new_dataset_name_prefix (str): Prefix for naming the output dataset files after PCA application.
        - metadata_columns (int, optional): Number of initial columns in the DataFrame to be treated as metadata and excluded from PCA. Defaults to 2.
        - min_features (int, optional): Minimum number of features required to perform PCA on a bin. Defaults to 8.
        - step_size (float, optional): Step size for binning the ppm range. Defaults to 0.005.
        - overlap (float, optional): Overlap between consecutive bins in the ppm range. Defaults to 0.0025.
        - n_components (int, optional): Number of principal components to retain during PCA. Defaults to 2.

        This class processes a given spectroscopic dataset by dividing it into bins based on specified step size and overlap, then applies PCA to each bin while retaining a specified number of principal components. The results, including PCA-transformed data and metadata about the transformation, are saved to disk.
        """
        self.file_path = file_path
        self.root = root
        self.new_dataset_name_prefix = new_dataset_name_prefix
        self.metadata_columns = metadata_columns
        self.min_features = min_features
        self.step_size = step_size
        self.overlap = overlap
        self.n_components = n_components

    def _save_data(self, new_dataset_name, data_with_labels, ppm_feature_names_dict, variance_explained_dict):
        """
        Saves the processed data and metadata to files.

        Parameters:
        - new_dataset_name (str): Name for the output dataset.
        - data_with_labels (pd.DataFrame): The PCA-transformed data with labels.
        - ppm_feature_names_dict (dict): Mapping of bin indices to feature names.
        - variance_explained_dict (dict): Explained variance for each PCA component.
        """
        new_name = f"{self.root}/{new_dataset_name}__bin{str(self.step_size)[2:]}_overlap{str(self.overlap)[2:]}"
        data_with_labels.to_csv(new_name + ".csv", index=False)
        with open(new_name + ".bins_to_names", 'wb') as f:
            pickle.dump(ppm_feature_names_dict, f)
        with open(new_name + ".bins_variance", 'wb') as f:
            pickle.dump(variance_explained_dict, f)

    def apply_dim_reduction(self):
        """
        Applies PCA dimensionality reduction to the dataset by binning based on the specified ppm range, step size, and overlap. Saves the PCA-transformed data along with metadata.
        """
        df = pd.read_csv(self.file_path)
        metadata = df.iloc[:, :self.metadata_columns]
        df = df.iloc[:, self.metadata_columns:]

        pca_components_dict, ppm_feature_names_dict, variance_explained_dict = {}, {}, {}
        index, start_ppm = 0, df.columns.astype(float).max()
        last_column_ppm = df.columns.astype(float).min()

        while start_ppm >= last_column_ppm:
            end_ppm = start_ppm - self.step_size
            bin_df = df.loc[:, (df.columns.astype(float) <= start_ppm) & (df.columns.astype(float) >= end_ppm)]
            if bin_df.shape[1] >= self.min_features:
                scaler = StandardScaler()
                bin_df_scaled = scaler.fit_transform(bin_df)

                pca = PCA(n_components=self.n_components)
                pca_components = pca.fit_transform(bin_df_scaled)

                pca_components_dict[f'bin_{index}'] = pca_components[:, 0]
                ppm_feature_names_dict[index] = bin_df.columns.tolist()
                variance_explained_dict[index] = pca.explained_variance_ratio_[0]

                index += 1
                start_ppm = end_ppm + self.overlap
            else:
                print(f"Not enough columns in the range {start_ppm} to {end_ppm}, combining with the next bin")
                start_ppm -= self.step_size  # Skip this bin if not enough features

        data_with_labels = pd.concat([metadata.reset_index(drop=True), pd.DataFrame.from_dict(pca_components_dict)], axis=1)

        # new_dataset_name = f"{self.new_dataset_name_prefix}__PCA"
        self._save_data(self.new_dataset_name_prefix, data_with_labels, ppm_feature_names_dict, variance_explained_dict)
        print(f"PCA Binning and data saving completed for {self.file_path}")
