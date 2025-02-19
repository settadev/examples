from setta.tasks.fns import SettaInMemoryFn
import base64
import io
from PIL import Image

$SETTA_GENERATED_PYTHON

def base64_to_pil(base64_string: str) -> Image.Image:
    # Remove data URL prefix if present
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Convert base64 to PIL Image
    img_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_bytes))

    return image


def pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


def drawing_to_img(project):
    x = base64_to_pil(project["input_image"]["drawing"])

    image_tensor = stream.preprocess_image(x)

    for _ in range(stream.batch_size - 1):
        stream(image=image_tensor)

    output_image = stream(image=image_tensor)

    output = pil_to_base64(output_image)
    return [
        {
            "name": "output_" + $stream_prepared$version,
            "type": "img",
            "value": output,
        }
    ]

fn = SettaInMemoryFn(fn=drawing_to_img)