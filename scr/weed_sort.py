import os
import random
import shutil

def copy_random_images(src_dir, dst_dir, num_images=500, extensions={'.jpg', '.jpeg', '.png', '.tif', '.tiff'}):
    """
    Randomly selects `num_images` from `src_dir` and copies them to `dst_dir`.

    Args:
        src_dir (str): Source directory containing images.
        dst_dir (str): Destination directory where selected images will be copied.
        num_images (int): Number of images to copy.
        extensions (set): Allowed image file extensions.
    """
    # Create destination folder if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Get list of image files in the source directory
    all_images = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower() in extensions
    ]

    if len(all_images) < num_images:
        raise ValueError(f"Not enough images in source directory. Found {len(all_images)} images.")

    # Randomly select the images
    selected_images = random.sample(all_images, num_images)

    # Copy the selected images to the destination folder
    for img_name in selected_images:
        shutil.copy(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

    print(f"✅ Successfully copied {num_images} images to {dst_dir}")

# Example usage
if __name__ == "__main__":
    source_directory = "/mnt/research-projects/j/jlgage/RawUAVData01/data/images"
    destination_directory = "/mnt/research-projects/j/jlgage/RawUAVData01/data/weeds"
    copy_random_images(source_directory, destination_directory, num_images=500)




