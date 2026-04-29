# WHITE-Net : White matter HyperIntensities Tissue Extraction using deep learning Network

WMH Segmentation tool using 3D ResUnet architecture. 

## INSTALLATION

You can install the package via `pip`:

### Install via pip 

```
pip install git+https://github.com/cathalacamille/WHITE-Net.git
```

### Manual installation
Alternatively, you can clone the repository and install the package manually:

```
git clone https://github.com/cathalacamille/WHITE-Net.git
cd WHITE-Net
pip install -r requirements.txt
pip install .
```

## DATA PREPARATION 

Ensure your data is preprocessed with skull stripping before using WHITE-Net. 
You can use tools like FSL's BET or SPM unified segmentation (keeping tissue probability maps c1 c2 and c3 of the GM, WM and cerebro-spinal
fluid (CSF)).

Data should be in nifti format.

## USAGE 

After installation, you can use the tool from the command line or within a Python script. Below are examples of how to use it.

### Running the Command-Line Tool

To run the tool, use the following command:
```
apply_whitenet /path/to/flair 
```

* flair_directory: The path to the directory containing skull-stripped FLAIR images (required).

### Using in Python Scripts
You can also import the package and use its functions directly in your Python code:

```
from WHITE-Net import apply_whitenet

# Example usage within a Python script
apply_whitenet(flair_directory="/path/to/flair")
```

### Outputs

The ouput WMH mask will be located in the FLAIR directory with the following name :
whitenet_FLAIR_WMH_[flair_name] 


## DEPENDENCIES

The tool requires the following Python packages:

* nibabel
* numpy
* glob2
* scikit-image
* torch
* argparse
* python-math
* scipy

These dependencies will be installed automatically when using the installation methods described above.



