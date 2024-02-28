import os
import argparse
import pandas as pd
from WaveletDenoise import WaveletDenoise
from PCABinning import PCABinning


def main(args):
    # Use arguments passed from command line
    save_path = args.save_path
    metadata_columns = args.metadata_columns
    unique_id_col = args.unique_id_col
    class_label_col = args.class_label_col
    wavelets = args.wavelets.split(',')  # Expecting a comma-separated list of wavelets

    # Load the dataset to be denoised
    cpmg = pd.read_csv(args.dataset_path)

    # Remove outliers
    subs_to_remove = args.subs_to_remove.split(',')  # Expecting a comma-separated list of subject IDs
    o_cpmg = cpmg.drop(cpmg[cpmg[unique_id_col].isin(subs_to_remove)].index)

    # Wavelet Denoising
    denoiser = WaveletDenoise(
        df=o_cpmg,
        metadata_columns=metadata_columns,
        unique_id_col=unique_id_col,
        class_label_col=class_label_col,
        save_path=save_path,
        wavelets=wavelets,
        save_data=True,
        show_graph=True
    )
    denoiser.process_and_visualize()

    # PCA Binning for each wavelet-transformed dataset
    for wav in wavelets:
        filename = f'cpmg_{wav}.csv'
        print(f"Processing file: {filename}")

        file_path = os.path.join(save_path, filename)
        base_name = os.path.splitext(filename)[0]
        # new_dataset_name_prefix = base_name + "__pca_reduction"

        pca_binner = PCABinning(
            file_path=file_path,
            root=save_path,
            new_dataset_name_prefix=base_name,
            step_size=args.step_size,
            overlap=args.overlap,
            n_components=args.n_components
        )

        # Apply PCA dimensionality reduction and save the results
        pca_binner.apply_dim_reduction()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess spectroscopic data using Wavelet Denoising and PCA Binning.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (CSV).")
    parser.add_argument("--save_path", type=str, required=True, help="Directory where the preprocessed files will be saved.")
    parser.add_argument("--metadata_columns", type=int, default=2, help="Number of metadata columns in the dataset.")
    parser.add_argument("--unique_id_col", type=str, default="Subject ID", help="Name of the column used as a unique identifier.")
    parser.add_argument("--class_label_col", type=str, default="Vasoplegia", help="Name of the column containing class labels.")
    parser.add_argument("--wavelets", type=str, default="db1", help="Comma-separated list of wavelets to use for denoising.")
    parser.add_argument("--subs_to_remove", type=str, default="", help="Comma-separated list of subject IDs to remove as outliers.")
    parser.add_argument("--step_size", type=float, default=0.005, help="Step size for the PCA binning process.")
    parser.add_argument("--overlap", type=float, default=0.0025, help="Overlap between bins for the PCA binning process.")
    parser.add_argument("--n_components", type=int, default=2, help="Number of PCA components to retain.")

    args = parser.parse_args()
    main(args)
