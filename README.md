# WHITE-Net : White matter HyperIntensities Tissue Extraction using deep learning Network

WMH Segmentation tool using 3D ResUnet architecture. 2 models are available : 1 using only FLAIR contrast and 1 using FLAIR+T1-weighted as input.

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
apply_whitenet /path/to/flair [/path/to/t1w] 
```

* flair_directory: The path to the directory containing skull-stripped FLAIR images (required).
* t1w_directory: The path to the directory containing skull-stripped T1w images  (optional).

### Using in Python Scripts
You can also import the package and use its functions directly in your Python code:

```
from WHITE-Net import apply_whitenet

# Example usage within a Python script
apply_whitenet(flair_directory="/path/to/flair", t1w_directory="/path/to/t1w")
```

### Outputs

If only FLAIR is provided as input, the ouput masks will be located in the FLAIR directory with the following names :
whitenet_FLAIR_WM_[flair_name] for WM mask and whitenet_FLAIR_WMH_[flair_name] for WMH mask

If FLAIR and T1-weighted are provided as inputs, the ouput masks will be located in the FLAIR directory with the following names :
whitenet_FLAIR_T1w_WM_[flair_name] for WM mask and whitenet_FLAIR_T1w_WMH_[flair_name] for WMH mask
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



