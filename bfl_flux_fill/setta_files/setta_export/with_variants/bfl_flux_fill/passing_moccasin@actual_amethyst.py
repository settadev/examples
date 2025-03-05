from setta.tasks.fns import SettaInMemoryFn
from $base64_utils$import_path import (
    base64_to_pil,
    convert_transparent_to_black,
    pil_to_base64,
)

$SETTA_GENERATED_PYTHON

pipe = pipe.to("cuda")


def _fn(p):
    layers = p["input"]["layers"]
    image = base64_to_pil(layers[0])
    width, height = image.size
    mask = layers[1]
    mask = convert_transparent_to_black(mask)
    output = pipe(
        **pipe_args,
        image=image,
        mask_image=mask,
        width=width,
        height=height,
    ).images[0]
    output = pil_to_base64(output)
    return [{"name": "output", "type": "img", "value": output}]


fn = SettaInMemoryFn(fn=_fn, dependencies=[])
