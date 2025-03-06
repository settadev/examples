from setta.tasks.fns import SettaInMemoryFn
from $base64_utils$import_path import (
    base64_to_pil,
    convert_transparent_to_black,
    pil_to_base64,
)
from PIL import Image

$SETTA_GENERATED_PYTHON

inpainter["model"] = inpainter["model"].to("cuda")

first_call = True


def inpaint_or_outpaint(layers, prompt, output_name):
    global first_call
    if first_call:
        first_call = False
        image = Image.open("input_imgs/bfl_example.png")
    else:
        layers = layers[1:]
        image = base64_to_pil(layers[0])
        width, height = image.size
        mask = layers[1]
        mask = convert_transparent_to_black(mask)
        image = inpainter["model"](
            **inpainter["args"],
            prompt=prompt,
            image=image,
            mask_image=mask,
            width=width,
            height=height,
        ).images[0]

    output_path = f"output_imgs/{output_name}.png"
    image.save(output_path)
    return [
        {
            "name": output_name,
            "type": "img",
            "path": output_path,
            "value": pil_to_base64(image),
        }
    ]


def outpaint_fn(p):
    drawing = p["input"]["drawing"]
    return inpaint_or_outpaint(drawing, p["inpaint_prompt"]["text"], "outpainted")


fn2 = SettaInMemoryFn(fn=outpaint_fn, dependencies=["input['drawing']"])
