
# SpectraFlow

This project contains the model and experimental implementation as detailed in the publication:

"SpectraFlow: A Novel Feature Selection Framework for Overcoming Challenges in 1D NMR Spectroscopy"

The SpectraFlow framework synergistically combines denoising, PCA-binning, sequential attention, and an MLP model to address challenges specific to feature selection on 1D NMR data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Before you begin, ensure you have Conda installed on your system. You can install either Miniconda or Anaconda:

- [Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda installation guide](https://www.anaconda.com/products/individual)

### Setting Up the Environment

#### Option 1: Using `spectraflow_env.yml`

To recreate the Conda environment with all necessary dependencies, follow these steps:

1. Open your terminal.
2. Navigate to the project directory.
3. Run the following command to create the environment:

```bash
conda env create -f spectraflow_env.yml
```

4. Once the environment is created, activate it:

```bash
conda activate spectraflow
```

#### Option 2: Using `requirements.txt`

If you prefer to use a virtual environment and install the core libraries without Conda, follow these steps:

1. Ensure you have Python and `pip` installed on your system.
2. Create a virtual environment:

```bash
python -m venv myenv
```

3. Activate the virtual environment:

- On Windows:
```cmd
myenv\Scripts\activate
```

- On Unix or MacOS:
```bash
source myenv/bin/activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Project
After setting up the environment, you can run the project using the provided shell scripts:

Running Preprocessing

Before running run_preprocess.sh, ensure it is executable:

```bash
chmod +x run_preprocess.sh
```

Then, you can run the preprocessing script which should not take longer than a minute on a normal machine:

```bash
./run_preprocess.sh
```
Remember to check and modify any variables or paths inside the script as necessary for your environment.

Running SpectraFlow Analysis

For run_spectraflow.sh, it is recommended to use a cluster of CPU machines due to the intensive computational resources required. First, make the script executable:

```bash
chmod +x run_spectraflow.sh
```

Afterwards, modify any variables or paths in the script according to your cluster environment and input data. Then execute:

```bash
./run_spectraflow.sh
```

## Documentation
View code [documentation](https://adigoryl.github.io/SpectraFlow/Docs/build/html/py-modindex.html)

## Code as a part of the publication
View publication [documentation](https://adigoryl.github.io/SpectraFlow/Docs/Oxford_Bioinformatics__SpectraFlow_v1.pdf)

## Contributing

If you would like to contribute to the project, please read `CONTRIBUTING.md` (if you have this file) for details on our code of conduct, and the process for submitting pull requests to us.

## Code Authors

- **Adrian Wesek** - [adigoryl](https://github.com/adigoryl)

See also the list of [contributors](https://github.com/yourproject/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

