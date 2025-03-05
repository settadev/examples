from setta.tasks.fns import SettaInMemoryFn
from $base64_utils$import_path import (
    base64_to_pil,
    convert_transparent_to_black,
    pil_to_base64,
)

$SETTA_GENERATED_PYTHON

inpainter["model"] = inpainter["model"].to("cuda")
redux["adapter"] = redux["adapter"].to("cuda")
redux["model"] = redux["model"].to("cuda")


def inpaint_fn(p):
    layers = p["input"]["drawing"][1:]
    image = base64_to_pil(layers[0])
    width, height = image.size
    mask = layers[1]
    mask = convert_transparent_to_black(mask)
    output = inpainter["model"](
        **inpainter["args"],
        prompt=p["prompt"]["text"],
        image=image,
        mask_image=mask,
        width=width,
        height=height,
    ).images[0]
    output = pil_to_base64(output)
    return [{"name": "inpainted", "type": "img", "value": output}]


prev_inpainted_image = None


def redux_fn(p):
    global prev_inpainted_image
    image = p["inpainted"]["image"]
    if image == prev_inpainted_image:
        return
    prev_inpainted_image = image
    image = base64_to_pil(image)
    adapter_output = redux["adapter"](image)
    images = redux["model"](**redux["args"], **adapter_output).images
    output = pil_to_base64(images[0])
    return [{"name": "variation", "type": "img", "value": output}]


fn = SettaInMemoryFn(fn=inpaint_fn, dependencies=[])
fn2 = SettaInMemoryFn(fn=redux_fn, dependencies=["inpainted['image']"])
