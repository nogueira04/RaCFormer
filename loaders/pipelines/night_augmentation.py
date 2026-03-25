"""
Night augmentation transforms for simulating nighttime driving conditions.

This module provides augmentation transforms to simulate night conditions
on daytime images for improving model robustness to lighting variations.
"""

import numpy as np
from numpy import random
from mmdet.datasets.builder import PIPELINES

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@PIPELINES.register_module()
class SimulateNight:
    """
    Transform daytime images to simulate night conditions.

    Applies a sequence of transformations to create realistic night-like images:
    1. Gamma correction (darken shadows more than highlights)
    2. Brightness reduction
    3. Contrast reduction
    4. Gaussian noise (simulates high ISO sensor noise)
    5. Optional: color temperature shift (orange streetlights)
    6. Optional: vignetting (darken corners)
    7. Optional: headlight gradient (brighten lower-center)
    8. Optional: random bright spots (simulate streetlights)
    9. Optional: preserve bright regions (adaptive darkening)

    All augmentation parameters are shared across all cameras in a single sample
    to maintain consistency, but each sample gets different random parameters.

    Args:
        prob (float): Probability of applying augmentation. Default: 0.3
        brightness_range (tuple): Min/max brightness multiplier. Default: (0.2, 0.5)
        contrast_range (tuple): Min/max contrast multiplier. Default: (0.5, 0.8)
        noise_std_range (tuple): Min/max noise standard deviation. Default: (5, 20)
        gamma_range (tuple): Min/max gamma value (>1 darkens). Default: (1.5, 2.5)
        color_shift (bool): Whether to apply streetlight color shift. Default: False
        color_shift_strength (tuple): Min/max strength of color shift. Default: (0.1, 0.3)

        # Enhanced realism features
        vignette (bool): Apply vignetting (darken corners). Default: False
        vignette_strength (tuple): Min/max vignette strength. Default: (0.3, 0.6)

        headlight_gradient (bool): Add headlight illumination gradient. Default: False
        headlight_strength (tuple): Min/max headlight brightness boost. Default: (0.2, 0.5)
        headlight_height (float): How far up the image headlights reach (0-1). Default: 0.4

        random_bright_spots (bool): Add random bright spots (streetlights). Default: False
        num_bright_spots (tuple): Min/max number of bright spots. Default: (3, 8)
        spot_brightness (tuple): Min/max spot brightness. Default: (150, 255)
        spot_size_range (tuple): Min/max spot radius in pixels. Default: (10, 40)

        preserve_bright (bool): Preserve already-bright regions. Default: False
        bright_threshold (int): Pixels above this are considered bright. Default: 200
        bright_preserve_factor (float): How much to preserve bright pixels (0-1). Default: 0.7
    """

    def __init__(self,
                 prob=0.3,
                 brightness_range=(0.2, 0.5),
                 contrast_range=(0.5, 0.8),
                 noise_std_range=(5, 20),
                 gamma_range=(1.5, 2.5),
                 color_shift=False,
                 color_shift_strength=(0.1, 0.3),
                 # Enhanced realism features
                 vignette=False,
                 vignette_strength=(0.3, 0.6),
                 headlight_gradient=False,
                 headlight_strength=(0.2, 0.5),
                 headlight_height=0.4,
                 random_bright_spots=False,
                 num_bright_spots=(3, 8),
                 spot_brightness=(150, 255),
                 spot_size_range=(10, 40),
                 preserve_bright=False,
                 bright_threshold=200,
                 bright_preserve_factor=0.7):

        assert 0.0 <= prob <= 1.0, f"prob must be in [0, 1], got {prob}"
        assert len(brightness_range) == 2 and brightness_range[0] <= brightness_range[1]
        assert len(contrast_range) == 2 and contrast_range[0] <= contrast_range[1]
        assert len(noise_std_range) == 2 and noise_std_range[0] <= noise_std_range[1]
        assert len(gamma_range) == 2 and gamma_range[0] <= gamma_range[1]

        self.prob = prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.color_shift = color_shift
        self.color_shift_strength = color_shift_strength

        # Enhanced features
        self.vignette = vignette
        self.vignette_strength = vignette_strength

        self.headlight_gradient = headlight_gradient
        self.headlight_strength = headlight_strength
        self.headlight_height = headlight_height

        self.random_bright_spots = random_bright_spots
        self.num_bright_spots = num_bright_spots
        self.spot_brightness = spot_brightness
        self.spot_size_range = spot_size_range

        self.preserve_bright = preserve_bright
        self.bright_threshold = bright_threshold
        self.bright_preserve_factor = bright_preserve_factor

    def _sample_parameters(self):
        """Sample random augmentation parameters for this sample."""
        params = {
            'gamma': random.uniform(*self.gamma_range),
            'brightness': random.uniform(*self.brightness_range),
            'contrast': random.uniform(*self.contrast_range),
            'noise_std': random.uniform(*self.noise_std_range),
            'color_strength': random.uniform(*self.color_shift_strength) if self.color_shift else 0,
            'vignette_strength': random.uniform(*self.vignette_strength) if self.vignette else 0,
            'headlight_strength': random.uniform(*self.headlight_strength) if self.headlight_gradient else 0,
            'num_spots': random.randint(self.num_bright_spots[0], self.num_bright_spots[1] + 1) if self.random_bright_spots else 0,
        }
        return params

    def _create_vignette_mask(self, h, w, strength):
        """
        Create a vignette mask that darkens corners.

        Args:
            h (int): Image height
            w (int): Image width
            strength (float): Vignette strength (0-1)

        Returns:
            np.ndarray: Vignette mask [H, W] with values in [1-strength, 1]
        """
        # Create coordinate grids normalized to [-1, 1]
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)

        # Compute distance from center (elliptical for non-square images)
        dist = np.sqrt(X**2 + Y**2)

        # Normalize to [0, 1] range (corners will be ~1.414)
        dist = dist / np.sqrt(2)

        # Create smooth falloff using cosine
        # At center: mask = 1, at corners: mask = 1 - strength
        mask = 1.0 - strength * (dist ** 1.5)
        mask = np.clip(mask, 1.0 - strength, 1.0)

        return mask.astype(np.float32)

    def _create_headlight_gradient(self, h, w, strength, height_ratio):
        """
        Create a gradient mask simulating ego vehicle headlight illumination.

        The headlights illuminate the lower-center portion of the image.

        Args:
            h (int): Image height
            w (int): Image width
            strength (float): Maximum brightness boost
            height_ratio (float): How far up the headlights reach (0-1)

        Returns:
            np.ndarray: Headlight mask [H, W] with values in [0, strength]
        """
        # Create coordinate grids
        y = np.linspace(0, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)

        # Vertical gradient: strongest at bottom, fading upward
        # Only affects bottom portion of image
        y_mask = np.clip((Y - (1 - height_ratio)) / height_ratio, 0, 1)

        # Horizontal: strongest in center, fading to sides
        # Use a wide gaussian-like falloff
        x_mask = np.exp(-2 * X**2)

        # Combine
        mask = y_mask * x_mask * strength

        return mask.astype(np.float32)

    def _add_bright_spots(self, img, num_spots, brightness_range, size_range):
        """
        Add random bright spots to simulate streetlights.

        Args:
            img (np.ndarray): Input image [H, W, 3]
            num_spots (int): Number of bright spots to add
            brightness_range (tuple): Min/max spot brightness
            size_range (tuple): Min/max spot radius

        Returns:
            np.ndarray: Image with bright spots added
        """
        h, w = img.shape[:2]
        img = img.astype(np.float32)

        for _ in range(num_spots):
            # Random position - bias toward upper portion (where streetlights are)
            cx = random.randint(0, w)
            cy = random.randint(0, int(h * 0.7))  # Upper 70% of image

            # Random size and brightness
            radius = random.randint(size_range[0], size_range[1])
            brightness = random.randint(brightness_range[0], brightness_range[1])

            # Create soft circular spot using distance from center
            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

            # Soft falloff
            spot_mask = np.clip(1 - dist / radius, 0, 1) ** 2

            # Add spot with slight color variation (warm light)
            color_variation = random.uniform(0.9, 1.1, 3)
            # Streetlights are often warm (more red/yellow)
            color_variation[2] *= 1.1  # Boost red
            color_variation[0] *= 0.9  # Reduce blue

            for c in range(3):
                img[..., c] = img[..., c] + spot_mask * brightness * color_variation[c]

        return img

    def _apply_night_transform(self, img, params):
        """
        Apply night simulation to a single image.

        Args:
            img (np.ndarray): Input image, uint8 [H, W, 3] BGR format
            params (dict): Augmentation parameters

        Returns:
            np.ndarray: Transformed image, uint8 [H, W, 3]
        """
        h, w = img.shape[:2]

        # Store original for bright region preservation
        if self.preserve_bright:
            original = img.copy()
            bright_mask = (img.mean(axis=2) > self.bright_threshold).astype(np.float32)
            # Smooth the mask slightly
            if HAS_SCIPY:
                bright_mask = gaussian_filter(bright_mask, sigma=3)

        # Convert to float for processing
        img = img.astype(np.float32)

        # 1. Gamma correction: darkens shadows more than highlights
        # Formula: img = 255 * (img/255)^gamma
        img = 255.0 * np.power(img / 255.0, params['gamma'])

        # 2. Brightness reduction
        img = img * params['brightness']

        # 3. Contrast reduction
        # Reduce contrast around the mean
        mean = img.mean()
        img = (img - mean) * params['contrast'] + mean

        # 4. Apply vignette (before noise, after basic darkening)
        if params['vignette_strength'] > 0:
            vignette_mask = self._create_vignette_mask(h, w, params['vignette_strength'])
            img = img * vignette_mask[..., np.newaxis]

        # 5. Apply headlight gradient
        if params['headlight_strength'] > 0:
            headlight_mask = self._create_headlight_gradient(
                h, w, params['headlight_strength'], self.headlight_height
            )
            # Add brightness (additive, scaled by current brightness to look natural)
            img = img + headlight_mask[..., np.newaxis] * 100  # Boost by up to 100 levels

        # 6. Add random bright spots (streetlights)
        if params['num_spots'] > 0:
            img = self._add_bright_spots(
                img, params['num_spots'],
                self.spot_brightness, self.spot_size_range
            )

        # 7. Gaussian noise (simulates high ISO sensor noise)
        if params['noise_std'] > 0:
            noise = random.normal(0, params['noise_std'], img.shape)
            img = img + noise

        # 8. Color temperature shift (optional - orange streetlight effect)
        # In BGR format: boost red (channel 2), reduce blue (channel 0)
        if params['color_strength'] > 0:
            strength = params['color_strength']
            # Boost red channel
            img[..., 2] = img[..., 2] * (1.0 + strength)
            # Slightly boost green
            img[..., 1] = img[..., 1] * (1.0 + strength * 0.3)
            # Reduce blue channel
            img[..., 0] = img[..., 0] * (1.0 - strength * 0.5)

        # 9. Preserve bright regions from original image
        if self.preserve_bright:
            # Blend original bright regions back in
            bright_mask_3d = bright_mask[..., np.newaxis]
            preserve_amount = self.bright_preserve_factor
            # For bright pixels, blend between darkened and original
            img = img * (1 - bright_mask_3d * preserve_amount) + \
                  original.astype(np.float32) * bright_mask_3d * preserve_amount

        # 10. Clip to valid range and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def __call__(self, results):
        """
        Apply night simulation to all images in the results dict.

        Args:
            results (dict): Result dict from loading pipeline.
                Expected to contain 'img' key with list of images.

        Returns:
            dict: Updated result dict with transformed images.
        """
        if random.random() > self.prob:
            return results

        if 'img' not in results:
            return results

        imgs = results['img']
        if not isinstance(imgs, list):
            imgs = [imgs]
            was_list = False
        else:
            was_list = True

        if len(imgs) == 0:
            return results

        # Sample parameters once for all cameras in this sample
        params = self._sample_parameters()

        # Apply transform to all images
        transformed_imgs = []
        for img in imgs:
            if img is not None:
                transformed_imgs.append(self._apply_night_transform(img, params))
            else:
                transformed_imgs.append(None)

        if was_list:
            results['img'] = transformed_imgs
        else:
            results['img'] = transformed_imgs[0]

        # Optionally store that night augmentation was applied
        results['night_augmented'] = True
        results['night_aug_params'] = params

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'brightness_range={self.brightness_range}, '
        repr_str += f'contrast_range={self.contrast_range}, '
        repr_str += f'noise_std_range={self.noise_std_range}, '
        repr_str += f'gamma_range={self.gamma_range}, '
        repr_str += f'color_shift={self.color_shift}, '
        repr_str += f'vignette={self.vignette}, '
        repr_str += f'headlight_gradient={self.headlight_gradient}, '
        repr_str += f'random_bright_spots={self.random_bright_spots}, '
        repr_str += f'preserve_bright={self.preserve_bright})'
        return repr_str


def apply_simulate_night(img,
                         gamma=2.0,
                         brightness=0.35,
                         contrast=0.65,
                         noise_std=10,
                         color_shift_strength=0.0,
                         vignette_strength=0.0,
                         headlight_strength=0.0,
                         headlight_height=0.4,
                         num_bright_spots=0,
                         spot_brightness=(150, 255),
                         spot_size_range=(10, 40),
                         preserve_bright=False,
                         bright_threshold=200,
                         bright_preserve_factor=0.7):
    """
    Standalone function to apply night simulation to a single image.

    Useful for visualization and testing without the pipeline infrastructure.

    Args:
        img (np.ndarray): Input image, uint8 [H, W, 3] BGR format
        gamma (float): Gamma correction value (>1 darkens). Default: 2.0
        brightness (float): Brightness multiplier (0-1). Default: 0.35
        contrast (float): Contrast multiplier (0-1). Default: 0.65
        noise_std (float): Noise standard deviation. Default: 10
        color_shift_strength (float): Color shift strength (0-1). Default: 0.0
        vignette_strength (float): Vignette strength (0-1). Default: 0.0
        headlight_strength (float): Headlight gradient strength. Default: 0.0
        headlight_height (float): How far up headlights reach (0-1). Default: 0.4
        num_bright_spots (int): Number of bright spots to add. Default: 0
        spot_brightness (tuple): Min/max spot brightness. Default: (150, 255)
        spot_size_range (tuple): Min/max spot radius. Default: (10, 40)
        preserve_bright (bool): Preserve bright regions. Default: False
        bright_threshold (int): Brightness threshold. Default: 200
        bright_preserve_factor (float): Preservation factor. Default: 0.7

    Returns:
        np.ndarray: Transformed image, uint8 [H, W, 3]
    """
    h, w = img.shape[:2]

    # Store original for bright region preservation
    if preserve_bright:
        original = img.copy()
        bright_mask = (img.mean(axis=2) > bright_threshold).astype(np.float32)
        if HAS_SCIPY:
            bright_mask = gaussian_filter(bright_mask, sigma=3)

    # Convert to float for processing
    img = img.astype(np.float32)

    # 1. Gamma correction
    img = 255.0 * np.power(img / 255.0, gamma)

    # 2. Brightness reduction
    img = img * brightness

    # 3. Contrast reduction
    mean = img.mean()
    img = (img - mean) * contrast + mean

    # 4. Vignette
    if vignette_strength > 0:
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2) / np.sqrt(2)
        vignette_mask = 1.0 - vignette_strength * (dist ** 1.5)
        vignette_mask = np.clip(vignette_mask, 1.0 - vignette_strength, 1.0)
        img = img * vignette_mask[..., np.newaxis]

    # 5. Headlight gradient
    if headlight_strength > 0:
        y = np.linspace(0, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)
        y_mask = np.clip((Y - (1 - headlight_height)) / headlight_height, 0, 1)
        x_mask = np.exp(-2 * X**2)
        headlight_mask = y_mask * x_mask * headlight_strength
        img = img + headlight_mask[..., np.newaxis] * 100

    # 6. Bright spots
    if num_bright_spots > 0:
        for _ in range(num_bright_spots):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, int(h * 0.7))
            radius = np.random.randint(spot_size_range[0], spot_size_range[1])
            spot_bright = np.random.randint(spot_brightness[0], spot_brightness[1])

            y_coords, x_coords = np.ogrid[:h, :w]
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            spot_mask = np.clip(1 - dist / radius, 0, 1) ** 2

            color_var = np.random.uniform(0.9, 1.1, 3)
            color_var[2] *= 1.1
            color_var[0] *= 0.9

            for c in range(3):
                img[..., c] = img[..., c] + spot_mask * spot_bright * color_var[c]

    # 7. Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, img.shape)
        img = img + noise

    # 8. Color temperature shift
    if color_shift_strength > 0:
        img[..., 2] = img[..., 2] * (1.0 + color_shift_strength)
        img[..., 1] = img[..., 1] * (1.0 + color_shift_strength * 0.3)
        img[..., 0] = img[..., 0] * (1.0 - color_shift_strength * 0.5)

    # 9. Preserve bright regions
    if preserve_bright:
        bright_mask_3d = bright_mask[..., np.newaxis]
        img = img * (1 - bright_mask_3d * bright_preserve_factor) + \
              original.astype(np.float32) * bright_mask_3d * bright_preserve_factor

    # 10. Clip and convert
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img
