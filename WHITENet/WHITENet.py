# Import necessary libraries
import numpy as np
import glob
from skimage import io
import nibabel as nib
import torch
import pkg_resources
import argparse
import os 

from .utils import *

def test(flair_directory):
    
    if os.path.isfile(flair_directory):
        # If it's a file and ends with .nii or .nii.gz, return it as a list
        if flair_directory.endswith('.nii') or flair_directory.endswith('.nii.gz'):
            data_FLAIR = [flair_directory]
        else:
            raise ValueError(f"The file {flair_directory} is not a .nii or .nii.gz file.")
    elif os.path.isdir(flair_directory):
        print(f"Processing FLAIR images in: {flair_directory}")
        data_FLAIR = sorted(glob.glob(f'{flair_directory}/*.nii') + glob.glob(f'{flair_directory}/*.nii.gz'))
    
    
    unet = UNet(n_in=1,n_class=1)
    model_path = pkg_resources.resource_filename(__name__, 'white_net_FLAIR.pt')
    unet.load_state_dict(torch.load(model_path,map_location=torch.device("cpu"))) 
    for i in range(len(data_FLAIR)):
        print(f"Processing {str(os.path.basename(data_FLAIR[i]))}")
        aff=nib.load(data_FLAIR[i])
        im=np.array((aff.get_fdata()))
        or_flair=nib.orientations.aff2axcodes(aff.affine)
        if desired_orientation!= or_flair:
            current_ornt = nib.orientations.axcodes2ornt(or_flair)
            desired_ornt = nib.orientations.axcodes2ornt(desired_orientation)
            transform = nib.orientations.ornt_transform(current_ornt, desired_ornt)
            # Apply the transformation to the image data
            im = nib.orientations.apply_orientation(im, transform)
        im_shape = im.shape
        mri_skullstripped=(im-np.min(im))/(np.max(im)-np.min(im))

        mri_preprocessed,ind_crop = crop_image(mri_skullstripped)

        if any(s > max_s for s, max_s in zip(mri_preprocessed.shape, (192,176,160))):
            mri_preprocessed,inverse_factors=resize_volume(mri_preprocessed, (192,176,160))
        else :
            inverse_factors= None
        original_shape=mri_preprocessed.shape
        mri=torch.tensor(zero_pad(mri_preprocessed,(192,176,160))).unsqueeze(0).unsqueeze(0)
        prob_map=unet(mri.float()).squeeze()
        output = adaptive_threshold_lesion_load(prob_map,thresh=0.5)
        output = torch.tensor(output)
        pred0= unpad(output,original_shape)

        if inverse_factors is not None:
            pred0 = (zoom(np.array(pred0),inverse_factors,order=3)>0.5).astype(int)

        wmh_pred = uncrop_image(pred0 ,ind_crop,im_shape)
        if desired_orientation!= or_flair:
            inverse_transform = nib.orientations.ornt_transform(desired_ornt, current_ornt)
            wmh_pred = (nib.orientations.apply_orientation(wmh_pred, inverse_transform)>0.5).astype(float)
            
        nifti_wmh = nib.Nifti1Image(wmh_pred,affine=aff.affine)
        # Save the NIfTI image to a file
        name= str(os.path.basename(data_FLAIR[i]))
        nib.save(nifti_wmh, str(os.path.dirname(data_FLAIR[i]))+'/whitenet_FLAIR_WMH_'+name)
        print(f"WMH mask saved in {str(os.path.dirname(data_FLAIR[i]))}")


   
def main():
    parser = argparse.ArgumentParser(description="Process FLAIR and optional T1w images.")
    
    # Required argument
    parser.add_argument('flair_directory', type=str, help="Path to the directory containing FLAIR images")
    
    # Optional argument
    parser.add_argument('t1w_directory', type=str, nargs='?', default=None, help="Path to the directory containing T1w images (optional)")

    args = parser.parse_args()
    
    test(args.flair_directory, args.t1w_directory)

if __name__ == "__main__":
    main()
