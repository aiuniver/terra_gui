import os
import math
import shutil
import zipfile
import tempfile

from PIL import Image


def autocrop_image_square(filepath: str, border: int = 0, min_size: int = 300):
    image = Image.open(filepath)

    # Get the bounding box
    bbox = image.getbbox()

    # Crop the image to the contents of the bounding box
    image = image.crop(bbox)

    # Determine the width and height of the cropped image
    (width, height) = image.size

    # Add border
    width += border * 2
    height += border * 2

    # Create a new image object for the output image
    square_size = height if height > width else width
    offset = math.ceil((min_size - square_size) / 2) if square_size < min_size else 0
    image_size = min_size if square_size < min_size else square_size
    cropped_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))

    # Paste the cropped image onto the new image
    cropped_image.paste(
        image,
        (
            border + offset + (int((square_size - width) / 2) if width < height else 0),
            border
            + offset
            + (int((square_size - height) / 2) if height < width else 0),
        ),
    )

    # Done!
    cropped_image.save(filepath)


def get_traceback_text(tb) -> list:
    output = [str(tb.tb_frame.f_code)]
    if tb.tb_next:
        output += get_traceback_text(tb.tb_next)
    return output


def unpack_model(filepath: str) -> dict:
    tmp_dir = tempfile.mkdtemp()
    model_tmp = os.path.join(tmp_dir, "model.zip")
    shutil.copy2(filepath, model_tmp)

    with zipfile.ZipFile(model_tmp, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    os.remove(model_tmp)

    return {
        "path": tmp_dir,
        "plan": os.path.join(tmp_dir, "plan.yaml"),
        "preview": os.path.join(tmp_dir, "preview.png"),
        "keras": os.path.join(tmp_dir, "keras.py"),
    }
