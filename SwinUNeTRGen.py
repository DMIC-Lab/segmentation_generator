import numpy as np
import torch

class SwinUNETRMaskGen:
    def __init__(self, weights_path, device='cuda'):
        """
        Initialize the MaskPreparer with the model weights and device.
        
        Parameters:
        weights_path (str): Path to the model weights.
        device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.weights = torch.load(weights_path)
        self.model = self.weights['model_arch']
        self.model.load_state_dict(self.weights['model_state_dict'])
        self.model.to(device)
        self.device = device
    
    def __call__(self, img, mask, roi):
        """
        Generate the mask for the given image and region of interest (ROI).
        
        Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Input mask.
        roi (numpy.ndarray): Region of interest.

        Returns:
        numpy.ndarray: Processed mask.
        numpy.ndarray: Model output.
        """
        self.max_x = 32
        self.max_y = 160
        self.max_z = 256

        min_x, min_y, min_z, max_x, max_y, max_z = self._get_roi_bounds(roi)
        reconstruct_pad = self._calculate_padding(img, min_x, max_x, min_y, max_y, min_z, max_z)
        
        img, mask = self._extract_roi(img, mask, roi, min_x, max_x, min_y, max_y, min_z, max_z)
        img, mask, reconstruct_pad = self._adjust_dimensions(img, mask, reconstruct_pad)
        img = self._normalize_image(img)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(img.unsqueeze(0).unsqueeze(0).to(self.device))
            out = torch.sigmoid(out)
        
        out_mask = self._threshold_output(out, thresh=0.65)
        out_mask = self._reconstruct_padded_output(out_mask, reconstruct_pad)
        
        return out_mask

    def _get_roi_bounds(self, roi):
        """
        Get the bounding coordinates of the region of interest (ROI).

        Parameters:
        roi (numpy.ndarray): Region of interest.

        Returns:
        tuple: Minimum and maximum coordinates (min_x, min_y, min_z, max_x, max_y, max_z).
        """
        min_coords = [np.min(np.where(roi == 1)[i]) for i in range(3)]
        max_coords = [np.max(np.where(roi == 1)[i]) for i in range(3)]
        
        # possibly multithread this to use 128 threads????
        for i in range(3):
            if ((roi.shape[i] - min_coords[i]) - (roi.shape[i] - max_coords[i]) ) < (32, 160, 256)[i]:
                if roi.shape[i] < (32, 160, 256)[i]:
                    min_coords[i] = 0
                    max_coords[i] = roi.shape[i] - roi.shape[i] % 32
                    if i == 1:
                        self.max_y = roi.shape[i] - roi.shape[i] % 32
                    elif i == 2:
                        self.max_z = roi.shape[i] - roi.shape[i] % 32
                    else:
                        raise ValueError("Invalid dimension size.")
                    
                    continue
                min_coords[i] = (roi.shape[i] - (32, 160, 256)[i]) // 2
                max_coords[i] = roi.shape[i] - ((roi.shape[i] - (32, 160, 256)[i]) // 2) 

        return (*min_coords, *max_coords)
    
    def _calculate_padding(self, img, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Calculate the padding needed to reconstruct the output to the original image size.

        Parameters:
        img (numpy.ndarray): Input image.
        min_x (int), max_x (int), min_y (int), max_y (int), min_z (int), max_z (int): ROI bounds.

        Returns:
        list: Padding values for reconstruction.
        """
        return [
            min_x, img.shape[0] - max_x,
            min_y, img.shape[1] - max_y,
            min_z, img.shape[2] - max_z
        ]
    
    def _extract_roi(self, img, mask, roi, min_x, max_x, min_y, max_y, min_z, max_z):
        """
        Extract the region of interest (ROI) from the image and mask.

        Parameters:
        img (numpy.ndarray): Input image.
        mask (numpy.ndarray): Input mask.
        roi (numpy.ndarray): Region of interest.
        min_x (int), max_x (int), min_y (int), max_y (int), min_z (int), max_z (int): ROI bounds.

        Returns:
        tuple: Cropped image and mask.
        """
        img[roi == 0] = 0
        return (
            img[min_x:max_x, min_y:max_y, min_z:max_z],
            mask[min_x:max_x, min_y:max_y, min_z:max_z]
        )
    
    def _adjust_dimensions(self, img, mask, reconstruct_pad):
        """
        Adjust the dimensions of the image and mask to be multiples of the model's requirements.

        Parameters:
        img (numpy.ndarray): Cropped image.
        mask (numpy.ndarray): Cropped mask.
        reconstruct_pad (list): Padding values for reconstruction.

        Returns:
        tuple: Adjusted image, mask, and updated padding values.
        """
        def calculate_removals(size, divisor):
            remove = size % divisor
            add = 0
            if remove % 2 != 0:
                remove //= 2
                add = 1
            else:
                remove //= 2
            return remove, add

        remove_x, add_x = calculate_removals(img.shape[0], self.max_x)
        remove_y, add_y = calculate_removals(img.shape[1], self.max_y)
        remove_z, add_z = calculate_removals(img.shape[2], self.max_z)

        reconstruct_pad[0] += remove_x
        reconstruct_pad[1] += remove_x + add_x
        reconstruct_pad[2] += remove_y
        reconstruct_pad[3] += remove_y + add_y
        reconstruct_pad[4] += remove_z
        reconstruct_pad[5] += remove_z + add_z

        img = img[remove_x:img.shape[0] - (remove_x + add_x),
                  remove_y:img.shape[1] - (remove_y + add_y),
                  remove_z:img.shape[2] - (remove_z + add_z)]

        mask = mask[remove_x:mask.shape[0] - (remove_x + add_x),
                    remove_y:mask.shape[1] - (remove_y + add_y),
                    remove_z:mask.shape[2] - (remove_z + add_z)]
        
        return img, mask, reconstruct_pad
    
    def _normalize_image(self, img):
        """
        Normalize the image.

        Parameters:
        img (numpy.ndarray): Input image.

        Returns:
        torch.Tensor: Normalized image.
        """
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        
        return (img - torch.mean(img)) / (torch.std(img) + 1e-6)
    
    def _threshold_output(self, out, thresh):
        """
        Apply a threshold to the model output.

        Parameters:
        out (torch.Tensor): Model output.
        thresh (float): Threshold value.

        Returns:
        numpy.ndarray: Thresholded output.
        """
        return (out > thresh)[0, 1].cpu().numpy()
    
    def _reconstruct_padded_output(self, out_mask, reconstruct_pad):
        """
        Reconstruct the padded output to match the original image size.

        Parameters:
        out_mask (numpy.ndarray): Thresholded output mask.
        reconstruct_pad (list): Padding values for reconstruction.

        Returns:
        numpy.ndarray: Reconstructed output mask.
        """
        for i in range(len(reconstruct_pad)):
            out_mask = np.pad(out_mask, (
                (reconstruct_pad[i] * (i == 0), reconstruct_pad[i] * (i == 1)),
                (reconstruct_pad[i] * (i == 2), reconstruct_pad[i] * (i == 3)),
                (reconstruct_pad[i] * (i == 4), reconstruct_pad[i] * (i == 5))
            ), mode='constant', constant_values=0)
        return out_mask

# Example usage:
# mask_preparer = MaskPreparer('path_to_weights.pth')
# out_mask, model_output = mask_preparer.prepare_mask(img, mask, roi, device='cuda')