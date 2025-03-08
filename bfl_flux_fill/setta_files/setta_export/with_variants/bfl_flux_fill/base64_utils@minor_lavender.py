import io
import base64
from PIL import Image


def convert_transparent_to_black(base64_image):
    image_data = base64.b64decode(base64_image)

    with io.BytesIO(image_data) as input_buffer:
        image = Image.open(input_buffer)

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        new_image = Image.new("RGBA", image.size, (0, 0, 0, 255))
        new_image.paste(image, (0, 0), image)
        return new_image.convert("RGB")


def base64_to_pil(base64_string: str) -> Image.Image:
    # Remove data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Convert base64 to PIL Image
    img_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_bytes))

    return image.convert("RGB")


def pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()
