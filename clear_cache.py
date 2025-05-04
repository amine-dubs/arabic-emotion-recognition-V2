#!/usr/bin/env python

"""
Cache clearing utility for Arabic Emotion Detection model

This script provides options to clear different parts of the cached data:
- All cache files
- Only model-related files
- Only evaluation results
- Selected combination of cache files
"""

import os
import argparse
import time

# Cache file paths
FEATURES_CACHE_PATH = 'resnet_features.npy'
DATA_CACHE_PATH = 'data_df.pkl'
INDICES_CACHE_PATH = 'selected_indices.npy'
SCALER_CACHE_PATH = 'scaler.joblib'
MODEL_CACHE_PATH = 'knn_final_model.joblib'
EVALUATION_RESULTS_PATH = 'evaluation_results.joblib'
PNG_FILES = ['confusion_matrix.png', 'accuracy_by_emotion.png', 
             'feature_importance.png', 'class_distribution.png']
MD_FILES = ['evaluation_report.md']

# Define file groups
ALL_CACHE_FILES = [
    FEATURES_CACHE_PATH, 
    DATA_CACHE_PATH, 
    INDICES_CACHE_PATH, 
    SCALER_CACHE_PATH, 
    MODEL_CACHE_PATH, 
    EVALUATION_RESULTS_PATH
] + PNG_FILES + MD_FILES

MODEL_FILES = [MODEL_CACHE_PATH, SCALER_CACHE_PATH]
EVALUATION_FILES = [EVALUATION_RESULTS_PATH] + PNG_FILES + MD_FILES
FEATURE_FILES = [INDICES_CACHE_PATH]

def delete_files(file_list, verbose=True):
    """Delete specified files and return count of deleted files."""
    deleted_count = 0
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                if verbose:
                    print(f"Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    return deleted_count

def main():
    parser = argparse.ArgumentParser(description='Clear cache files for Arabic Emotion Detection model')
    parser.add_argument('--all', action='store_true', help='Clear all cache files')
    parser.add_argument('--model', action='store_true', help='Clear model files')
    parser.add_argument('--evaluation', action='store_true', help='Clear evaluation results')
    parser.add_argument('--features', action='store_true', help='Clear feature selection files')
    parser.add_argument('--indices', action='store_true', help='Clear selected indices only')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    files_to_delete = []
    
    # Determine which files to delete
    if args.all:
        files_to_delete = ALL_CACHE_FILES
        if verbose:
            print("Clearing ALL cache files...")
    else:
        if args.model:
            files_to_delete.extend(MODEL_FILES)
            if verbose:
                print("Clearing model files...")
        
        if args.evaluation:
            files_to_delete.extend(EVALUATION_FILES)
            if verbose:
                print("Clearing evaluation results...")
        
        if args.features:
            files_to_delete.extend(FEATURE_FILES)
            if verbose:
                print("Clearing feature selection files...")
                
        if args.indices:
            files_to_delete.append(INDICES_CACHE_PATH)
            if verbose:
                print("Clearing selected indices...")
    
    # If no arguments provided, show help
    if len(files_to_delete) == 0:
        parser.print_help()
        return
    
    # Delete the files
    start_time = time.time()
    deleted_count = delete_files(files_to_delete, verbose)
    end_time = time.time()
    
    # Print summary
    if verbose:
        print(f"\nDeleted {deleted_count} files in {end_time - start_time:.2f} seconds.")
        print("Ready for retraining!")

if __name__ == "__main__":
    main()