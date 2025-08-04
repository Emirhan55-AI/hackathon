"""
Image processing utilities for Aura AI Platform
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Union, Optional
import base64


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes"""
    return Image.open(io.BytesIO(image_bytes))


def load_image_from_base64(base64_string: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    image_bytes = base64.b64decode(base64_string)
    return load_image_from_bytes(image_bytes)


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode()


def resize_image(
    image: Image.Image, 
    size: Tuple[int, int], 
    keep_aspect_ratio: bool = True
) -> Image.Image:
    """Resize image to target size"""
    if keep_aspect_ratio:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        # Pad with white background to exact size
        background = Image.new('RGB', size, (255, 255, 255))
        offset = ((size[0] - image.size[0]) // 2, (size[1] - image.size[1]) // 2)
        background.paste(image, offset)
        return background
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def normalize_image(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Normalize image to [0, 1] range"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """Denormalize image from [0, 1] to [0, 255] range"""
    return (image * 255.0).astype(np.uint8)


def crop_image(
    image: Image.Image, 
    bbox: Tuple[float, float, float, float]
) -> Image.Image:
    """Crop image using normalized bounding box coordinates"""
    x, y, width, height = bbox
    img_width, img_height = image.size
    
    left = int(x * img_width)
    top = int(y * img_height)
    right = int((x + width) * img_width)
    bottom = int((y + height) * img_height)
    
    return image.crop((left, top, right, bottom))


def apply_image_augmentation(
    image: Image.Image,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
    hue: Optional[float] = None
) -> Image.Image:
    """Apply image augmentation"""
    from PIL import ImageEnhance
    
    result = image.copy()
    
    if brightness is not None:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(brightness)
    
    if contrast is not None:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(contrast)
    
    if saturation is not None:
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(saturation)
    
    return result


def get_dominant_colors(
    image: Image.Image, 
    num_colors: int = 5
) -> list:
    """Extract dominant colors from image"""
    from sklearn.cluster import KMeans
    
    # Convert to RGB and resize for faster processing
    image = image.convert('RGB')
    image = image.resize((150, 150))
    
    # Convert to numpy array and reshape
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    
    return colors.tolist()
