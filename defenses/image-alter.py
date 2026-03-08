import kagglehub
import os
import numpy as np
from PIL import Image

# Download the mammals dataset from KaggleHub
path = kagglehub.dataset_download("asaniczka/mammals-image-classification-dataset-45-animals")

# Create the root output folder for all defensive versions
target_folder = os.path.expanduser("~/.cache/defensive_images")
os.makedirs(target_folder, exist_ok=True)

# Create subfolders for each of the 5 transformation types
transform_names = [
    "1_crop_enlarge",
    "2_rotated",
    "3_jpeg_compressed",
    "4_bit_depth_reduced",
    "5_combined"
]
for name in transform_names:
    os.makedirs(os.path.join(target_folder, name), exist_ok=True)

first_folder = os.path.join(path, os.listdir(path)[0])  # /mammals
animal_categories = os.listdir(first_folder)

for animal_category in animal_categories:
    modified_animal_category = os.path.join(first_folder, animal_category)

    # Create per-transform subfolders for each animal category
    for name in transform_names:
        os.makedirs(os.path.join(target_folder, name, animal_category), exist_ok=True)

    for image_file in os.listdir(modified_animal_category):
        modified_image_path = os.path.join(modified_animal_category, image_file)
        filename, _ = os.path.splitext(image_file)

        try:
            image = Image.open(modified_image_path).convert("RGB")

            # ------------------------------------------------------------------
            # Transform 1: Crop center 32x32 patch from a 64x64 region, enlarge back
            # ------------------------------------------------------------------
            w, h = image.size
            cx, cy = w // 2, h // 2
            # Crop a 64x64 region centered on the image
            region_64 = image.crop((cx - 32, cy - 32, cx + 32, cy + 32))
            # Crop the center 32x32 from that region
            region_32 = region_64.crop((16, 16, 48, 48))
            # Enlarge back to original size
            t1 = region_32.resize((w, h), Image.NEAREST)
            t1.save(os.path.join(target_folder, "1_crop_enlarge", animal_category, filename + ".png"))

            # ------------------------------------------------------------------
            # Transform 2: Rotate by a fixed angle (45 degrees)
            # ------------------------------------------------------------------
            t2 = image.rotate(45, expand=False)
            t2.save(os.path.join(target_folder, "2_rotated", animal_category, filename + ".png"))

            # ------------------------------------------------------------------
            # Transform 3: Convert PNG -> JPEG with compression (quality=30)
            # ------------------------------------------------------------------
            t3_path = os.path.join(target_folder, "3_jpeg_compressed", animal_category, filename + ".jpg")
            image.save(t3_path, format="JPEG", quality=30)

            # ------------------------------------------------------------------
            # Transform 4: Bit-depth reduction (posterize to 2 bits per channel)
            # ------------------------------------------------------------------
            arr = np.array(image)
            # Keep only the top 2 bits of each 8-bit channel (values quantized to multiples of 64)
            arr_reduced = (arr >> 6) << 6
            t4 = Image.fromarray(arr_reduced.astype(np.uint8))
            t4.save(os.path.join(target_folder, "4_bit_depth_reduced", animal_category, filename + ".png"))

            # ------------------------------------------------------------------
            # Transform 5: Combined (all four transforms applied in sequence)
            # ------------------------------------------------------------------
            t5 = t1.rotate(45, expand=False)
            arr5 = np.array(t5)
            arr5 = (arr5 >> 6) << 6
            t5 = Image.fromarray(arr5.astype(np.uint8))
            t5_path = os.path.join(target_folder, "5_combined", animal_category, filename + ".jpg")
            t5.save(t5_path, format="JPEG", quality=30)

        except (OSError, Exception) as e:
            print(f"Error processing {modified_image_path}: {e}")
            continue

print(f"All defensive transforms saved to: {target_folder}")