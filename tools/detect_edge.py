import cv2
from pathlib import Path
from tqdm import tqdm


data_dir = Path("/svl/u/briannlz/magicpony_sdxl_canny/val")
image_suffix = "depth.png"
edge_suffix = "canny.png"



if __name__ == "__main__":
    image_paths = list(data_dir.glob("**/*" + image_suffix))
    for image_path in tqdm(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 10, 200)
        edge_path = image_path.parent / image_path.name.replace(image_suffix, edge_suffix)
        cv2.imwrite(str(edge_path), edges)
