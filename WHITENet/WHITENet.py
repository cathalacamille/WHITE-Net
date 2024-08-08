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

def test(flair_directory, t1w_directory=None):
    
    flair_directory = os.path.abspath(flair_directory)
    print(f"Processing FLAIR images in: {flair_directory}")
    
    data_FLAIR = sorted(glob.glob(flair_directory))
    if t1w_directory:
        data_T1w = sorted(glob.glob(t1w_directory))
    else : 
        data_T1w = None 
    desired_orientation = ('P', 'S', 'R')    
    if  data_T1w == None : 
        unet = UNet(n_in=1,n_class=2)
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
                inverse_factors=[None]
            original_shape=mri_preprocessed.shape
            mri=torch.tensor(zero_pad(mri_preprocessed,(192,176,160))).unsqueeze(0).unsqueeze(0)
            output=(unet(mri.float())>0.5).int()

            pred0= unpad(output[0,0,:,:,:],original_shape)
            pred1= unpad(output[0,1,:,:,:],original_shape)

            if inverse_factors!=None:
                pred0 = (zoom(np.array(pred0),inverse_factors,order=3)>0.5).astype(int)
                pred1 = (zoom(np.array(pred1),inverse_factors,order=3)>0.5).astype(int)

            wm_pred = uncrop_image(pred0 ,ind_crop,im_shape)
            wmh_pred = uncrop_image(pred1 ,ind_crop,im_shape)
            if desired_orientation!= or_flair:
                inverse_transform = nib.orientations.ornt_transform(desired_ornt, current_ornt)
                wm_pred = (nib.orientations.apply_orientation(wm_pred, inverse_transform)>0.5).astype(float)
                wmh_pred = (nib.orientations.apply_orientation(wmh_pred, inverse_transform)>0.5).astype(float)
                
            nifti_wm = nib.Nifti1Image(wm_pred,affine=aff.affine)  
            nifti_wmh = nib.Nifti1Image(wmh_pred,affine=aff.affine)
            # Save the NIfTI image to a file
            name= str(os.path.basename(data_FLAIR[i]))
            nib.save(nifti_wm, str(os.path.dirname(data_FLAIR[i]))+'/whitenet_FLAIR_WM_'+name)
            nib.save(nifti_wmh, str(os.path.dirname(data_FLAIR[i]))+'/whitenet_FLAIR_WMH_'+name)
            print(f"WMH and WM masks saved in {str(os.path.dirname(data_FLAIR[i]))}")


    else:
        unet = UNet(n_in=2,n_class=2)
        model_path = pkg_resources.resource_filename(__name__, 'white_net_FLAIR_T1w.pt')
        unet.load_state_dict(torch.load(model_path,map_location=torch.device("cpu"))) 
        for i in range(len(data_FLAIR)):
            print(f"Processing {str(os.path.basename(data_FLAIR[i]))}")
            aff=nib.load(data_FLAIR[i])
            im=np.array(aff.get_fdata())
            t1=np.array(nib.load(data_T1w[i]).get_fdata())
            or_flair=nib.orientations.aff2axcodes(aff.affine)
            if desired_orientation!= or_flair:
                current_ornt = nib.orientations.axcodes2ornt(or_flair)
                desired_ornt = nib.orientations.axcodes2ornt(desired_orientation)
                transform = nib.orientations.ornt_transform(current_ornt, desired_ornt)
                # Apply the transformation to the image data
                im = nib.orientations.apply_orientation(im, transform)
                t1 = nib.orientations.apply_orientation(t1, transform)
            im_shape = im.shape

            mri_skullstripped=(im-np.min(im))/(np.max(im)-np.min(im))
            t1_skullstripped=(t1-np.min(t1))/(np.max(t1)-np.min(t1))
            mri_preprocessed, t1_preprocessed,ind_crop = crop_image(mri_skullstripped, t1_skullstripped)

            if any(s > max_s for s, max_s in zip(mri_preprocessed.shape, (192,176,160))):

                mri_preprocessed,inverse_factors=resize_volume(mri_preprocessed, (192,176,160))
                t1_preprocessed,_=resize_volume(t1_preprocessed, (192,176,160))
            else :
                inverse_factors=None
            original_shape=mri_preprocessed.shape
            mri=torch.tensor([zero_pad(mri_preprocessed,(192,176,160)),zero_pad(t1_preprocessed,(192,176,160))]).unsqueeze(0)
            output=(unet(mri.float())>0.5).int()


            pred0= unpad(output[0,0,:,:,:],original_shape)
            pred1= unpad(output[0,1,:,:,:],original_shape)

            if inverse_factors!=None:
                pred0 = (zoom(np.array(pred0),inverse_factors,order=3)>0.5).astype(int)
                pred1 = (zoom(np.array(pred1),inverse_factors,order=3)>0.5).astype(int)

            wm_pred = uncrop_image(pred0 ,ind_crop,im_shape)
            wmh_pred = uncrop_image(pred1 ,ind_crop,im_shape)
            if desired_orientation!= or_flair:
                inverse_transform = nib.orientations.ornt_transform(desired_ornt, current_ornt)
                wm_pred = (nib.orientations.apply_orientation(wm_pred, inverse_transform)>0.5).astype(float)
                wmh_pred = (nib.orientations.apply_orientation(wmh_pred, inverse_transform)>0.5).astype(float)


            nifti_wm = nib.Nifti1Image(wm_pred,affine=aff.affine)  
            nifti_wmh = nib.Nifti1Image(wmh_pred,affine=aff.affine)
            # Save the NIfTI image to a file
            name= str(os.path.basename(data_FLAIR[i]))
            nib.save(nifti_wm, str(os.path.dirname(data_FLAIR[i]))+'/whitenet_FLAIR_T1w_WM_'+name)
            nib.save(nifti_wmh, str(os.path.dirname(data_FLAIR[i]))+'/whitenet_FLAIR_T1w_WMH_'+name)
            print(f"WMH and WM masks saved in {str(os.path.dirname(data_FLAIR[i]))}")
            
    
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