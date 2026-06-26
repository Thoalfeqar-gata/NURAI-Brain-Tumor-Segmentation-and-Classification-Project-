import cv2, numpy as np
from PIL import Image

class CLAHETransform:
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img):
        # img: PIL Image (RGB) or grayscale
        arr = np.array(img)
        if arr.ndim == 2:  # grayscale
            out = self.clahe.apply(arr)
        else:
            # arr is RGB (HxWx3). Convert to LAB, apply CLAHE to L channel.
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)

