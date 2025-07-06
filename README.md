# MOSTLY AI Prize Competition Submission



**Competition Entry for [The MOSTLY AI Prize](https://www.mostlyaiprize.com/) - Stage 2**

This repository contains an optimized synthetic data generation solution for both **FLAT DATA** and **SEQUENTIAL DATA** challenges, capable of generating high-fidelity privacy-safe synthetic data that meets competition requirements.


## 🚀 Quick Start

### Prerequisites

**Recommended AWS EC2 Instance Types:**
- **GPU runs**: `g5.2xlarge` 

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mostly-ai-competition

# Install dependencies
pip install -r requirements.txt
pip install uv
uv pip install -U "mostlyai-engine[gpu]"
pip install psycopg2-binary
```

In case the installation fails, all the requirements and their dependencies can be found in the `requirements_with_dependency.txt` file.


Otherwise, you can use the Dockerfile to build the image and run the container by running:
```bash
docker build -t mostlyai-engine-2 .
docker run --gpus=all -it mostlyai-engine-2 -v data:/data
```

### Competition Usage

The main competition script `run_best_config.py` accepts a CSV training dataset and generates a synthetic dataset of identical size and structure.

#### For SEQUENTIAL DATA Challenge

```bash
python run_best_config.py \
    --input-file /data/sequential_training.csv \
    --model-type sequential \
    --sample-size 20000 \
    --workspace /data/workspace_sequential \
    --name sequential_submission
```



>In the sequential case, the selected column configuration is saved in the `config/column_types.json` file and loaded automatically if the column names are not the same, those must be updated in the file.

>Consider that the input and output directory /data is referred to the one mounted in the container, if you are running without the container, you need to change the path to the input and output files.

> By default, the run will remove the workspace directory if already exists a previous one, consider to save the results in a different directory every run.

### Command Line Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--input-file` | Path to input CSV training dataset | - | ✅ |
| `--model-type` | Model type: `tabular` or `sequential` | `sequential` | ❌ |
| `--sample-size` | Number of synthetic samples to generate | `20000` | ❌ |
| `--workspace` | Workspace directory for intermediate files | `/data/workspace_best` | ❌ |
| `--name` | Experiment name for output files | `best_config_flat` | ❌ |

## 📊 Optimized Hyperparameters

This submission uses extensively tuned hyperparameters optimized for competition metrics:

### SEQUENTIAL DATA Tabular Model
- **Architecture**: Medium model with optimized embeddings
- **Training**: Weight decay 0.049, LR factor 0.35, dropout 0.45
- **Performance**: Optimized for overall accuracy while maintaining privacy


## 🔧 Technical Details

### Dependencies

See `requirements.txt` for pinned versions and `requirements_with_dependency.txt` for all the dependencies and their versions.


## 📁 Output Files

After successful execution, the following files are generated:

```
{workspace}/
├── SyntheticData/          # Main synthetic dataset (Parquet format)
├── synthetic_data_{name}.csv  # Competition-ready CSV output
├── results_{name}.json     # Detailed metrics and parameters
└── ModelStore/            # Trained model artifacts
```


## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

