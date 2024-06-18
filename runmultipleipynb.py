from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read, write
import os

# List of notebook files to execute
notebook_files = [
    "02_com_data_augmentation_batch_32_layers_[256, 512, 1024, 1024].ipynb",
    "02_com_data_augmentation_batch_64_layers_[256,512,1024,1024].ipynb",
    "02_com_data_augmentation_batch_128_layers_[256,512,1024,1024].ipynb",
    "02_com_data_augmentation_batch_256_layers_[256,512,1024,1024].ipynb",
    "main_resnet50_unfreezed_100_batch_64_layers_[1024, 512, 256, 128].ipynb",
    "main_resnet50_unfreezed_100_batch_128_layers_[1024, 512, 256, 128].ipynb",
    "main_resnet50_unfreezed_150_batch_64_layers_[1024, 512, 256, 128].ipynb",
    "main_resnet50_unfreezed_150_batch_128_layers_[1024, 512, 256, 128].ipynb",
    "main_sem_data_augmentation_batch_32_layers_[256, 512, 1024, 1024].ipynb",
    "main_sem_data_augmentation_batch_64_layers_[256, 512, 1024, 1024].ipynb",
    "main_sem_data_augmentation_batch_128_layers_[256, 512, 1024, 1024].ipynb",
    "main_sem_data_augmentation_batch_256_layers_[256, 512, 1024, 1024].ipynb"
]

# Directory to save executed notebooks
output_dir = "executed_notebooks_with_outputs"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to execute a notebook and save with outputs
def execute_notebook_with_outputs(notebook_file):
    with open(notebook_file, "r", encoding="utf-8") as f:
        notebook = read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=None, kernel_name='python3')
    ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_file)}})
    
    # Save executed notebook with outputs in the output directory
    output_file = os.path.join(output_dir, os.path.basename(notebook_file))
    with open(output_file, "w", encoding="utf-8") as f:
        write(notebook, f)

    print(f"Notebook {notebook_file} executed successfully and saved to {output_file} with outputs.\n")

# Execute all notebooks in the list
for file in notebook_files:
    print(f"Executing notebook: {file}")
    execute_notebook_with_outputs(file)
