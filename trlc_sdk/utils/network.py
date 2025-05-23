import base64
import io
from PIL import Image

def rle_to_single_mask(rle):
    import pycocotools.mask as mask_util
    if isinstance(rle["counts"], str):
        rle = {**rle, "counts": rle["counts"].encode("utf-8")}
    mask = mask_util.decode(rle)
    # Remove the last dimension if present (HxWx1 -> HxW)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    return mask

def encode_file(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image(image: Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')