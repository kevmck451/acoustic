import imageio.v2 as imageio
import os
import re

def generate_gif(input_folder, output_folder, name, frame_duration):

    def sort_filenames(filenames):
        """ Sort filenames with numbers in a human-readable order """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(filenames, key=alphanum_key)

    # File extension of the images (e.g., jpg, png)
    image_extension = 'png'

    # Collecting file names of images
    images = [img for img in os.listdir(input_folder) if img.endswith(image_extension)]
    images = sort_filenames(images)  # Sort the images by name in a human-readable order

    # Creating a list of images for the GIF
    frames = [imageio.imread(os.path.join(input_folder, img)) for img in images]

    output = f'{output_folder}/{name}'

    # Saving the images as a GIF
    imageio.mimsave(output, frames, duration=frame_duration)
