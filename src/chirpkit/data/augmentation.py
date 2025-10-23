import numpy as np
import librosa
import torch
import random

class InsectAudioAugmenter:
    """Enhanced audio augmentation for insect sounds"""
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def time_stretch(self, spectrogram: np.ndarray, rate: float = None) -> np.ndarray:
        """Time stretch augmentation on spectrogram with fixed output size"""
        if rate is None:
            rate = random.uniform(0.8, 1.2)
        
        # Keep original time dimensions to maintain batch consistency
        time_steps = spectrogram.shape[1]
        
        # Apply time stretch but resample back to original size
        new_time_steps = int(time_steps * rate)
        
        if new_time_steps > 0:
            # First stretch
            indices = np.linspace(0, time_steps - 1, new_time_steps)
            indices = np.round(indices).astype(int)
            stretched = spectrogram[:, indices]
            
            # Then resample back to original size
            final_indices = np.linspace(0, new_time_steps - 1, time_steps)
            final_indices = np.round(final_indices).astype(int)
            final_indices = np.clip(final_indices, 0, new_time_steps - 1)
            return stretched[:, final_indices]
        return spectrogram
    
    def frequency_shift(self, spectrogram: np.ndarray, shift_max: int = 5) -> np.ndarray:
        """Shift frequencies up or down"""
        shift = random.randint(-shift_max, shift_max)
        if shift == 0:
            return spectrogram
        
        shifted = np.zeros_like(spectrogram)
        if shift > 0:
            shifted[shift:] = spectrogram[:-shift]
        else:
            shifted[:shift] = spectrogram[-shift:]
        return shifted
    
    def add_noise(self, spectrogram: np.ndarray, noise_factor: float = None) -> np.ndarray:
        """Add Gaussian noise to spectrogram"""
        if noise_factor is None:
            noise_factor = random.uniform(0.001, 0.01)
        
        noise = np.random.normal(0, noise_factor, spectrogram.shape)
        return spectrogram + noise
    
    def frequency_mask(self, spectrogram: np.ndarray, num_masks: int = None, mask_param: int = 15) -> np.ndarray:
        """SpecAugment frequency masking"""
        if num_masks is None:
            num_masks = random.randint(1, 2)
            
        spec = spectrogram.copy()
        for _ in range(num_masks):
            f = random.randint(1, mask_param)
            f0 = random.randint(0, max(1, spec.shape[0] - f))
            spec[f0:f0 + f, :] = spec.mean()
        return spec
    
    def time_mask(self, spectrogram: np.ndarray, num_masks: int = None, mask_param: int = 20) -> np.ndarray:
        """SpecAugment time masking"""
        if num_masks is None:
            num_masks = random.randint(1, 2)
            
        spec = spectrogram.copy()
        for _ in range(num_masks):
            t = random.randint(1, mask_param)
            t0 = random.randint(0, max(1, spec.shape[1] - t))
            spec[:, t0:t0 + t] = spec.mean()
        return spec
    
    def mixup_spectrogram(self, spec1: np.ndarray, spec2: np.ndarray, alpha: float = 0.4) -> tuple:
        """Mixup augmentation for spectrograms"""
        lam = np.random.beta(alpha, alpha)
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        return mixed_spec, lam
    
    def apply_random_augmentation(self, spectrogram: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Apply random augmentation with given probability"""
        if random.random() > prob:
            return spectrogram
        
        # Choose random augmentation
        augmentations = [
            self.time_stretch,
            self.frequency_shift, 
            self.add_noise,
            self.frequency_mask,
            self.time_mask
        ]
        
        aug_func = random.choice(augmentations)
        return aug_func(spectrogram)

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper with augmentation"""
    def __init__(self, base_dataset, augmenter, augmentation_prob=0.5):
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.augmentation_prob = augmentation_prob
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        original_shape = x.shape
        
        # Apply augmentation to spectrogram
        if random.random() < self.augmentation_prob:
            # Convert tensor to numpy, augment, then back to tensor
            spec_np = x.squeeze().numpy()
            spec_aug = self.augmenter.apply_random_augmentation(spec_np, prob=1.0)
            x = torch.tensor(spec_aug, dtype=torch.float32).unsqueeze(0)
            
            # Ensure shape consistency
            if x.shape != original_shape:
                # Pad or crop to match original shape
                if x.shape[2] > original_shape[2]:
                    x = x[:, :, :original_shape[2]]
                elif x.shape[2] < original_shape[2]:
                    padding = original_shape[2] - x.shape[2]
                    x = torch.nn.functional.pad(x, (0, padding), mode='constant', value=0)
        
        return x, y
