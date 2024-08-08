import torch
import torch.nn as nn
from torch.nn.functional import relu
from math import ceil
from skimage.transform import resize
import numpy as np
from scipy.ndimage import zoom

def zero_pad(im,size=(256,256,256)):
    pad_im=np.zeros(size)
    diff_y=pad_im.shape[0]-im.shape[0]
    diff_x=pad_im.shape[1]-im.shape[1]
    diff_z=pad_im.shape[2]-im.shape[2]
    y1=diff_y//2
    y2=pad_im.shape[0]-ceil(diff_y/2)
    x1=diff_x//2
    x2=pad_im.shape[1]-ceil(diff_x/2)
    z1=diff_z//2
    z2=pad_im.shape[2]-ceil(diff_z/2)
    
    pad_im[y1:y2,x1:x2,z1:z2]=im
    return pad_im

def unpad(im, size=(256,256,256)):
    unpad_im = np.zeros(size)
    diff_y=im.shape[0]-unpad_im.shape[0]
    diff_x=im.shape[1]-unpad_im.shape[1]
    diff_z=im.shape[2]-unpad_im.shape[2]
    y1=diff_y//2
    y2=im.shape[0]-ceil(diff_y/2)
    x1=diff_x//2
    x2=im.shape[1]-ceil(diff_x/2)
    z1=diff_z//2
    z2=im.shape[2]-ceil(diff_z/2)
    unpad_im = im[y1:y2,x1:x2,z1:z2]
    return unpad_im

def crop_image(img_array, other_im=None, tol=0):
    mask_array = img_array > tol
    selected_indices = np.ix_(mask_array.any(axis=(1, 2)), mask_array.any(axis=(0, 2)), mask_array.any(axis=(0, 1)))
    cropped_images = img_array[selected_indices]
    if other_im is not None:
        cropped_other_im = other_im[selected_indices]
        return cropped_images, cropped_other_im, selected_indices
    return cropped_images, selected_indices


def uncrop_image(cropped_WMHS, original_indices, original_shape):
    uncropped_WMHS = np.zeros(original_shape)
    
    uncropped_WMHS[original_indices] = cropped_WMHS
    
    return uncropped_WMHS



def crop_image1(img_array, tol=0):
    mask_array = img_array > tol
    selected_indices = np.ix_(mask_array.any(axis=(1, 2)), mask_array.any(axis=(0, 2)), mask_array.any(axis=(0, 1)))
    cropped_images = img_array[selected_indices]
    return cropped_images, selected_indices


def resize_volume(volume, target_shape):
    # Calculate zoom factors for each dimension
    factors = [min(t/s, 1.0) for t, s in zip(target_shape, volume.shape)]
    factor=min(factors)
    inverse_factors = [1/f for f in factors]
    return zoom(volume, factors, order=3),inverse_factors  # Order=1 for linear interpolation




class UNet(nn.Module):
    def __init__(self,n_in, n_class, nb_kernels=16):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv3d(n_in, nb_kernels, kernel_size=3, padding=1)  # Adjusted padding
        self.e12 = nn.Conv3d(nb_kernels, nb_kernels, kernel_size=3, padding=1)  # Adjusted padding
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.e21 = nn.Conv3d(nb_kernels, nb_kernels * 2, kernel_size=3, padding=1)  # Adjusted padding
        self.e22 = nn.Conv3d(nb_kernels * 2, nb_kernels * 2, kernel_size=3, padding=1)  # Adjusted padding
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.e31 = nn.Conv3d(nb_kernels * 2, nb_kernels * 4, kernel_size=3, padding=1)  # Adjusted padding
        self.e32 = nn.Conv3d(nb_kernels * 4, nb_kernels * 4, kernel_size=3, padding=1)  # Adjusted padding
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.e41 = nn.Conv3d(nb_kernels * 4, nb_kernels * 8, kernel_size=3, padding=1)  # Adjusted padding
        self.e42 = nn.Conv3d(nb_kernels * 8, nb_kernels * 8, kernel_size=3, padding=1)  # Adjusted padding
        self.do4 = nn.Dropout(p=0.5)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.e51 = nn.Conv3d(nb_kernels * 8, nb_kernels * 16, kernel_size=3, padding=1)  # Adjusted padding
        self.e52 = nn.Conv3d(nb_kernels * 16, nb_kernels * 16, kernel_size=3, padding=1)  # Adjusted padding
        self.do5 = nn.Dropout(p=0.5)
        # Decoder
        self.upconv1 = nn.ConvTranspose3d(nb_kernels * 16, nb_kernels * 8, kernel_size=2, stride=2)  # Adjusted stride
        self.d11 = nn.Conv3d(nb_kernels * 16, nb_kernels * 8, kernel_size=3, padding=1)  # Adjusted padding
        self.d12 = nn.Conv3d(nb_kernels * 8, nb_kernels * 8, kernel_size=3, padding=1)  # Adjusted padding

        self.upconv2 = nn.ConvTranspose3d(nb_kernels * 8, nb_kernels * 4, kernel_size=2, stride=2)  # Adjusted stride
        self.d21 = nn.Conv3d(nb_kernels * 8, nb_kernels * 4, kernel_size=3, padding=1)  # Adjusted padding
        self.d22 = nn.Conv3d(nb_kernels * 4, nb_kernels * 4, kernel_size=3, padding=1)  # Adjusted padding

        self.upconv3 = nn.ConvTranspose3d(nb_kernels * 4, nb_kernels * 2, kernel_size=2, stride=2)  # Adjusted stride
        self.d31 = nn.Conv3d(nb_kernels * 4, nb_kernels * 2, kernel_size=3, padding=1)  # Adjusted padding
        self.d32 = nn.Conv3d(nb_kernels * 2, nb_kernels * 2, kernel_size=3, padding=1)  # Adjusted padding

        self.upconv4 = nn.ConvTranspose3d(nb_kernels * 2, nb_kernels, kernel_size=2, stride=2)  # Adjusted stride
        self.d41 = nn.Conv3d(nb_kernels * 2, nb_kernels, kernel_size=3, padding=1)  # Adjusted padding
        self.d42 = nn.Conv3d(nb_kernels, nb_kernels, kernel_size=3, padding=1)  # Adjusted padding

        # Output layer
        self.outconv = nn.Conv3d(nb_kernels, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = nn.functional.relu(self.e11(x))
        xe12 = nn.functional.relu(self.e12(xe11))+xe11
        xp1 = self.pool1(xe12)

        xe21 = nn.functional.relu(self.e21(xp1))
        xe22 = nn.functional.relu(self.e22(xe21))+xe21
        xp2 = self.pool2(xe22)

        xe31 = nn.functional.relu(self.e31(xp2))
        xe32 = nn.functional.relu(self.e32(xe31))+xe31
        xp3 = self.pool3(xe32)

        xe41 = nn.functional.relu(self.e41(xp3))
        xe42 = self.do4(nn.functional.relu(self.e42(xe41))+xe41)
        xp4 = self.pool4(xe42)

        xe51 = nn.functional.relu(self.e51(xp4))
        xe52 = self.do5(nn.functional.relu(self.e52(xe51))+xe51)

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = nn.functional.relu(self.d11(xu11))
        xd12 = nn.functional.relu(self.d12(xd11))+xd11

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = nn.functional.relu(self.d21(xu22))
        xd22 = nn.functional.relu(self.d22(xd21))+xd21

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = nn.functional.relu(self.d31(xu33))
        xd32 = nn.functional.relu(self.d32(xd31))+xd31

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = nn.functional.relu(self.d41(xu44))
        xd42 = nn.functional.relu(self.d42(xd41))+xd41

        # Output layer
        out = self.outconv(xd42)

        return out

