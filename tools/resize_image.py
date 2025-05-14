import os
from PIL import Image

"""Resize image and replace the original image with the resized one"""


input_dir = "data/mp_temp2"
output_dir = None
supported_extensions = ('rgb.png', 'mask.png', 'occlusion.png', 'keypoint.png', 'depth.png')
size = (256, 256)


if __name__ == "__main__":
    # If output_dir is not specified, overwrite images in place
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Traverse the input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(supported_extensions):
                input_path = os.path.join(root, file)

                # Determine the output path
                if output_dir:
                    # Maintain the directory structure
                    relative_path = os.path.relpath(root, input_dir)
                    target_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(target_dir, exist_ok=True)
                    output_path = os.path.join(target_dir, file)
                else:
                    # Overwrite the original image
                    output_path = input_path

                try:
                    with Image.open(input_path) as img:
                        # Optional: Preserve aspect ratio
                        # img.thumbnail(size, Image.ANTIALIAS)

                        # Resize the image to exact size (may distort if aspect ratio differs)
                        resized_img = img.resize(size)

                        # Save the resized image
                        # To maintain the original image quality, especially for JPEGs, you can specify quality
                        if img.mode in ("RGBA", "P"):
                            # For PNG and images with transparency
                            resized_img = resized_img.convert("RGB")
                            resized_img.save(output_path, quality=95)
                        else:
                            resized_img.save(output_path, quality=95)

                        print(f"Resized and saved: {output_path}")

                except Exception as e:
                    print(f"Failed to process {input_path}. Error: {e}")



