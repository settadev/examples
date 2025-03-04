from setta.tasks.fns import SettaInMemoryFn
import io
import base64
from PIL import Image

$SETTA_GENERATED_PYTHON

pipe = pipe.to("cuda")

def convert_transparent_to_black(base64_image):
    image_data = base64.b64decode(base64_image)

    with io.BytesIO(image_data) as input_buffer:
        image = Image.open(input_buffer)

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        new_image = Image.new("RGBA", image.size, (0, 0, 0, 255))
        new_image.paste(image, (0, 0), image)
        new_image = new_image.convert("RGB")
        output_buffer = io.BytesIO()
        new_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        result_base64 = base64.b64encode(output_buffer.read()).decode("utf-8")

    return result_base64


def _fn(p):
    layers = p["input"]["layers"]
    image = layers[0]
    mask = layers[1]
    mask = convert_transparent_to_black(mask)


fn = SettaInMemoryFn(fn=_fn)
