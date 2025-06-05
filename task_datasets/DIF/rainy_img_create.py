from pathlib import Path
import numpy as np
import cv2
import PIL.Image as Image
from skimage.transform import resize
from albumentations.augmentations import Rotate

def read_image(path: Path, convert: str='RGB') -> np.ndarray:
    image = Image.open(path).convert(convert)
    return np.array(image)

def save_image(image: np.ndarray, path: Path | str, convert: str='RGB'):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == '':
        path = path.with_suffix('.png')
    
    # rgba
    image = Image.fromarray(image).convert(convert)
    image.save(path, format='PNG', quality=100)
    
    print(f'save image {path.name}')
    
def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized_img = resize(
        image,
        output_shape=size,
        order=0,
        mode='reflect',
        cval=0,
        clip=True,
        preserve_range=True,
        anti_aliasing=True,
    )
    return resized_img


RAIN_DROP_PATHS = {
    "large": list(Path('/Data3/cao/ZiHanCao/datasets/RainMask/horizontal').glob('*.png')),
}
print(f'found {len(RAIN_DROP_PATHS["large"])} rainy drops for large rain')


def extend_image(image: np.ndarray, target_size: tuple[int, int], mode: str = 'wrap') -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 首先处理图像过大的情况
    if h > target_h or w > target_w:
        # 计算需要裁剪的区域
        start_h = (h - target_h) // 2 if h > target_h else 0
        start_w = (w - target_w) // 2 if w > target_w else 0
        end_h = start_h + min(h, target_h)
        end_w = start_w + min(w, target_w)
        image = image[start_h:end_h, start_w:end_w]
        h, w = image.shape[:2]
    
    # 然后处理图像过小的情况（原有逻辑）
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left
    
    extended = cv2.copyMakeBorder(
        image,
        top=max(0, top),
        bottom=max(0, bottom),
        left=max(0, left),
        right=max(0, right),
        borderType=getattr(cv2, f'BORDER_{mode.upper()}'),
        value=[0, 0, 0] if mode == 'constant' else None
    )
    
    return extended

def read_rain_drops(path: Path) -> np.ndarray:
    image = read_image(path)
    return image

def synthetic_rainy_pair(not_rainy: np.ndarray,
                         rain_drop_select_path: str | Path | None = None,
                         rain_drop_paths=RAIN_DROP_PATHS['large'],
                         size_matcher: str='pad',
                         unit8: bool=False) -> np.ndarray:
    if rain_drop_select_path is not None and isinstance(rain_drop_select_path, str):
        rain_drop_path = Path(rain_drop_select_path)
        assert rain_drop_path.exists(), f'rain drop path {rain_drop_path} not exists'
    else:
        rain_drop_path = np.random.choice(rain_drop_paths)
    
    # read the rain drop image
    rain_drop = read_rain_drops(rain_drop_path)
    
    rotate = Rotate(limit=(-10, 10), p=0.5)
    rain_drop = rotate(image=rain_drop)['image']
    if size_matcher == 'resize':
        rain_drop = resize_image(rain_drop, not_rainy.shape)
    elif size_matcher == 'pad':
        rain_drop = extend_image(rain_drop, not_rainy.shape[:2])
    else:
        raise ValueError(f'size_matcher must be in [resize, pad], but got {size_matcher}')
    
    rain_drop = rain_drop.astype(np.float32) / 255.
    not_rainy = not_rainy.astype(np.float32)
    
    #* ver 1
    # alpha = 0.8
    # rainy = not_rainy * (1. - alpha * rain_drop) + rain_drop * alpha
    
    #* ver 2
    y_not_rain, cr_not_rain, cb_not_rain = cv2.split(cv2.cvtColor(not_rainy, cv2.COLOR_RGB2YCrCb))
    y_rain = y_not_rain + cv2.cvtColor(rain_drop, cv2.COLOR_RGB2GRAY)
    rainy = cv2.cvtColor(cv2.merge([y_rain, cr_not_rain, cb_not_rain]), cv2.COLOR_YCrCb2RGB)
    
    rainy = np.clip(rainy * 255, 0, 255).astype(np.uint8) if unit8 else rainy.clip(0, 1)
    
    return rainy


def img_dir_to_rainy(image_dir: str, rain_type: str='large'):
    rain_drop_paths = RAIN_DROP_PATHS[rain_type]
    synthetic_dir = Path(image_dir).parent / 'synthetic_rainy'
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in Path(image_dir).glob('*'):
        img = read_image(image_path)
        
        # random choose an image drop
        rainy = synthetic_rainy_pair(img)
        
        save_path = synthetic_dir / image_path.name
        save_image(rainy, save_path)
        
if __name__ == '__main__':
    img_dir_to_rainy(image_dir='data/Rain100L/norain', rain_type='large')
