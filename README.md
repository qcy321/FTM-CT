# FTM-CT: A Fine-Tuning Method with Cross-Training Policy for Code Search 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9.13](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://www.python.org/downloads/release/python-3913/)

This repository contains the official implementation for the paper **"FTM-CT: A Fine-Tuning Method with Cross-Training Policy for Code Search".

FTM-CT is a novel fine-tuning strategy that enhances code search models by optimizing the contrastive learning process. It introduces an alternating training mechanism that uses a cache of historical negative samples encoded by a fixed, best-performing checkpoint. This approach provides more diverse and challenging negative signals, boosting the model's semantic discrimination ability while simultaneously reducing training costs.

## ✨ Key Features

* **📈 Enhanced Performance**: FTM-CT consistently improves the retrieval performance of various baseline models on code search tasks.
* **💸 Cost-Efficient**: By using a fixed encoder that doesn't require backpropagation, the `In-Policy` training stage significantly reduces computational overhead and training time.

## 🏁 Getting Started

Follow these steps to set up the project environment and run the experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/qcy321/FTM-CT.git
cd FTM-CT
```

### 2. Create Environment & Install Dependencies

It's recommended to use a virtual environment (like conda or venv).

```bash
# Create and activate a conda environment
conda create -n py39 python=3.9.13
conda activate py39

# Install required packages
conda install --file requirements.txt
```

### 3. Download Datasets

The provided scripts will automatically download and unzip the necessary datasets.

```bash
# Download the CodeSearchNet (CSN) dataset
cd dataset/CSN && bash run.sh && cd ../..

# Download the AdvTest dataset
cd dataset/AdvTest && bash run.sh && cd ../..

# Download the GPD dataset
cd dataset/GPD && bash run.sh && cd ../..
```

## 🛠️ Usage

### Step 1: Data Preprocessing

After downloading the datasets, you need to preprocess parts of the dataset. This only needs to be done once for each dataset.

```bash
# Preprocess AdvTest dataset
python data_processing.py --dataset "AdvTest" --num_processes 8

# Preprocess CosQA dataset
python data_processing.py --dataset "CosQA" --num_processes 8
```
*Note: Run the command for each dataset you intend to use.*

### Step 2: Fine-Tune and Evaluate

The `run.sh` script handles both the training and evaluation of the model.

```bash
# Run training and evaluation in the background
nohup bash run.sh &
```
## ⚙️ Project Configuration (`common.py`)

The `common.py` file is the central configuration hub for the project. It defines important constants, paths, and mappings that are used throughout the codebase. If you wish to extend the project by adding new models or datasets, this is the primary file you will need to modify.

Here is a breakdown of the key configurations:

* **`MODEL_CLASS_MAPPING` (Dictionary)**: If you need to run models other than those provided by this project, you must first register them here:
    After adding your model to the mapping, you must also implement its corresponding data processing logic in the `converter.py` file.

* **`DATA_CONFIG` (Dictionary)**: This dictionary manages the paths and filenames for all datasets used in the project.
    > **To add a new dataset**: Add a new key-value pair here, specifying the dataset's name, its folder path, and the necessary filenames in the correct order (codebase, train, test, valid).


## ⚙️ Configuring Experiments

Before running, you can configure your desired experiments by editing the variables at the bottom of the `run.sh` file.

```bash
# --- Configure your experiments in run.sh ---

# 1. Set datasets and their corresponding training epochs
declare -A dataset_epochs
dataset_epochs=(
    ["CSN"]=10
    #["AdvTest"]=4
)

# 2. Set programming languages
languages=("ruby" "javascript" "python")

# 3. Set cache queue sizes
sizes=(2048)

# 4. Select the models to run by commenting or uncommenting the `run_model` lines
for DATASET in "${!dataset_epochs[@]}"; do
    epoch=${dataset_epochs[$DATASET]}
    for la in "${languages[@]}"; do
        for size in "${sizes[@]}"; do
            #run_model "roberta" "roberta-base" "$la" $size "$DATASET" "avg" "$epoch"
            run_model "graphcodebert" "microsoft/graphcodebert-base" "$la" $size "$DATASET" "cls" "$epoch"
            # ... other models
        done
    done
done
```

#### Locating Results

After each successful experiment, the script automatically creates a dedicated directory to store all relevant outputs. You can find the results for each run in a path with the following structure:`[dataset]/[language]/[queue_size]/[strategy]/[epoch]/[checkpoint]/[model_name]/`

For example, the results for a `graphcodebert` model trained on the `CSN` dataset for the `python` language will be located in a directory like:
`CSN/python/2048/cross/10/best/graphcodebert/`

Inside this directory, you will find:
* `train.log`: The detailed training and evaluation log.
* `mrr_result.txt`: The final Mean Reciprocal Rank (MRR) evaluation scores.
* `saved_models/`: A directory containing the saved model checkpoints.

#### Important Notes

* **Sequential Execution**: Experiments are executed **sequentially**, one after another. The next task will only begin after the current one has completed successfully.
* **Automatic Skipping**: To prevent accidental overwrites, the script checks if an output directory for an experiment already exists. If it does, that experiment will be **skipped**. To re-run an experiment, you must first **manually delete its output directory**.
* **Process Cleanup**: The script is designed to manage all child processes. If you interrupt the script (e.g., with `Ctrl+C`), it will automatically terminate any ongoing training processes to prevent them from being orphaned.
* **Cache Management**: The data processing script creates cache files to accelerate data loading. If you modify the data conversion logic (`converter.py`), you **must delete the old cache files** for your changes to take effect. The cache file is stored in `saved_models/{language}/{model}_{dataset}_{running_mode}.pt`.
