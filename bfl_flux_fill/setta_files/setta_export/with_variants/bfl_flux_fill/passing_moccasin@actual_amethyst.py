from setta.tasks.fns import SettaInMemoryFn
from $base64_utils$import_path import (
    base64_to_pil,
    convert_transparent_to_black,
    pil_to_base64,
)

$SETTA_GENERATED_PYTHON

inpainter = inpainter.to("cuda")
adapter = adapter.to("cuda")
base_model = base_model.to("cuda")


def inpaint_fn(p):
    layers = p["input"]["layers"]
    image = base64_to_pil(layers[0])
    width, height = image.size
    mask = layers[1]
    mask = convert_transparent_to_black(mask)
    output = inpainter(
        **inpainter_args,
        image=image,
        mask_image=mask,
        width=width,
        height=height,
    ).images[0]
    output = pil_to_base64(output)
    return [{"name": "inpainted", "type": "img", "value": output}]


def redux_fn(p):
    image = p["inpainted"]["image"]
    image = base64_to_pil(image)
    adapter_output = adapter(image)
    images = base_model(**base_model_args, **adapter_output).images
    output = pil_to_base64(images[0])
    return [{"name": "variation", "type": "img", "value": output}]



fn = SettaInMemoryFn(fn=inpaint_fn, dependencies=[])
fn2 = SettaInMemoryFn(fn=redux_fn, dependencies=[])
