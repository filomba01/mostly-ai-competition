#!/usr/bin/env python3
"""
Simple script to run the best configuration for synthetic data generation.
Supports both tabular and sequential model types.
"""

import sys
from pathlib import Path
import pandas as pd
import torch
import json
import argparse
import shutil
import time
import dotenv
import os

dotenv.load_dotenv()

# add the src directory to the path
repo_root = Path(__file__).parent
possible_paths = [
    repo_root / "src",
    repo_root / "mostlyai",  
    repo_root,
]

# search for mostlyai in local directories
mostlyai_path = None
for path in possible_paths:
    if (path / "mostlyai").exists():
        mostlyai_path = path
        break
    elif (path / "__init__.py").exists() and "mostlyai" in str(path):
        mostlyai_path = path.parent
        break

if mostlyai_path:
    sys.path.insert(0, str(mostlyai_path))
    print(f"✅ Using local mostlyai source from: {mostlyai_path}")
else:
    print("⚠️  Could not find local mostlyai source, trying standard import...")

try:
    from mostlyai import engine
    print("✅ Successfully imported engine module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure mostlyai is installed: pip install -U mostlyai-engine")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run best configuration for synthetic data generation")
    
    parser.add_argument("--input-file", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--model-type", type=str, default="sequential", 
                       choices=["tabular", "sequential"],
                       help="Model type: tabular or sequential (default: sequential)")
    parser.add_argument("--sample-size", type=int, default=20000,
                       help="Number of synthetic samples to generate (default: 20000)")
    parser.add_argument("--workspace", type=str, default="/data/workspace_best",
                       help="Workspace directory (default: /data/workspace_best)")
    parser.add_argument("--name", type=str, default="best_config_flat",
                       help="Experiment name (default: best_config)")
    
    return parser.parse_args()


def load_column_types():
    """Load column types from config file"""
    config_path = Path("config/column_types.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        print("⚠️  Column types config not found, using auto-detection")
        return None


def run_best_config(args):
    """Run the best configuration pipeline"""
    
    # BEST HYPERPARAMETERS (from hyperparameter tuning results)
    BEST_PARAMS_SEQUENTIAL = {
        "seed": 46,
        "batch_size": 1024,
        "split_size": 0.9,
        "epochs": 1000,
        "max_sequence_window": 10,
        "model_size": "Medium",
        
        # Sampling parameters
        "sampling_temperature": 1.0,
        "sampling_top_p": 1.0,
        
        # Model architecture parameters
        "embedding_lower_bound": 19,
        "column_embedding_lower_bound": 8,
        "dropout_rate_history": 0.45,
        "dropout_rate_context": 0.25,
        "dropout_rate_regressor": 0.05,
        
        # Training parameters
        "weight_decay": 0.04993060823921497,
        "lr_factor": 0.35,
        "val_loss_patience": 10,
        "lr_patience": 2,
        
        # Generation parameters
        "enable_flexible_generation": False,
    }
    
    BEST_PARAMS_TABULAR = {
        "lr_factor": 0.1,
        "dropout_rate_regressor": 0.04818078201461677,
        "embedding_lower_bound": 17,
        "embedding_exponent": 0.3456641218404693,
        "initial_lr": 0.000200298699117235,
        "weight_decay": 0.003895813805322184,
        "batch_size": 512,
        "model_size": "Large",
        "seed": 0,
        "split_size": 0.95,
        "epochs": 1000,
        #"val_loss_patience": 10,
        # Sampling parameters
        "sampling_temperature": 1.0,
        "sampling_top_p": 1.0,
        # Generation parameters
        "enable_flexible_generation": False
    }
    
    print("🚀 Starting best configuration run...")
    print(f"📊 Model type: {args.model_type}")
    print(f"📁 Input file: {args.input_file}")
    print(f"🎯 Sample size: {args.sample_size}")
    print(f"💾 Workspace: {args.workspace}")
    
    # Load training data
    print("\n📖 Loading training data...")
    trn_df = pd.read_csv(args.input_file)
    print(f"✅ Loaded {len(trn_df)} rows with {len(trn_df.columns)} columns")
    print(f"📋 Columns: {list(trn_df.columns)}")
    
    if args.model_type == "sequential":
        BEST_PARAMS = BEST_PARAMS_SEQUENTIAL
        column_types = load_column_types()
    else:
        BEST_PARAMS = BEST_PARAMS_TABULAR
        column_types = None
        
    # Setup workspace
    ws = Path(args.workspace)
    if ws.exists():
        print(f"🗑️  Cleaning existing workspace: {ws}")
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize logging and random state
        engine.init_logging()
        engine.set_random_state(BEST_PARAMS["seed"])
        print(f"🎲 Set random seed to: {BEST_PARAMS['seed']}")
                
        # Step 1: Split data
        print("\n🔀 Splitting data...")
        engine.split(
            workspace_dir=ws,
            tgt_data=trn_df,
            model_type="TABULAR",
            tgt_encoding_types=column_types,
            trn_val_split=BEST_PARAMS["split_size"],
            tgt_context_key=(None if args.model_type == "tabular" else "group_id")
        )
        print("✅ Data split completed")
        
        # Step 2: Analyze
        print("\n🔍 Analyzing data...")
        engine.analyze(workspace_dir=ws)
        print("✅ Data analysis completed")
        
        # Step 3: Encode
        print("\n🔢 Encoding data...")
        engine.encode(workspace_dir=ws)
        print("✅ Data encoding completed")
        
        # Step 4: Train model
        print("\n🏋️  Training model...")
        
        # Build training parameters dict - only include parameters that exist
        train_params = {
            "model": f"MOSTLY_AI/{BEST_PARAMS['model_size']}",
            "workspace_dir": ws,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        # Add optional parameters only if they exist in BEST_PARAMS
        optional_train_params = [
            "batch_size", "max_sequence_window", "enable_flexible_generation", 
            "epochs", "val_loss_patience", "lr_patience", "lr_factor", 
            "embedding_lower_bound", "dropout_rate_history", "dropout_rate_context", 
            "dropout_rate_regressor", "column_embedding_lower_bound", "weight_decay",
            "initial_lr", "embedding_exponent"
        ]
        
        for param in optional_train_params:
            if param in BEST_PARAMS:
                # Handle special case where 'epochs' maps to 'max_epochs'
                if param == "epochs":
                    train_params["max_epochs"] = BEST_PARAMS[param]
                else:
                    train_params[param] = BEST_PARAMS[param]
        
        engine.train(**train_params)
        print("✅ Model training completed")
        
        # Step 5: Generate synthetic data
        print(f"\n🎭 Generating {args.sample_size} synthetic samples...")
        
        # Build generation parameters dict - only include parameters that exist
        gen_params = {
            "workspace_dir": ws,
            "sample_size": args.sample_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        
        # Add optional parameters only if they exist in BEST_PARAMS
        optional_gen_params = [
            "sampling_temperature", "sampling_top_p", "embedding_lower_bound",
            "dropout_rate_history", "dropout_rate_context", "dropout_rate_regressor",
            "column_embedding_lower_bound"
        ]
        
        for param in optional_gen_params:
            if param in BEST_PARAMS:
                gen_params[param] = BEST_PARAMS[param]
        
        engine.generate(**gen_params)
        print("✅ Synthetic data generation completed")
        
        # Step 6: Load and evaluate
        print("\n📊 Evaluating synthetic data quality...")
        generated_data = pd.read_parquet(ws / "SyntheticData")
        print(f"✅ Generated {len(generated_data)} synthetic samples")

        
        # Save synthetic data as CSV for easy access
        csv_file = ws / f"synthetic_data_{args.name}.csv"
        generated_data.to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Synthetic data: {csv_file}")
        print(f"   📁 Full workspace: {ws}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise


def main():
    args = parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        run_best_config(args)
        print("\n✅ Best configuration run completed successfully!")
    except Exception as e:
        print(f"\n❌ Run failed: {e}")
        raise


if __name__ == "__main__":
    main() 