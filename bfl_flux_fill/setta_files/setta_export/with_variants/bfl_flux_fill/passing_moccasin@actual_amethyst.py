from setta.tasks.fns import SettaInMemoryFn
from $base64_utils$import_path import (
    base64_to_pil,
    convert_transparent_to_black,
    pil_to_base64,
)
from PIL import Image
import torch

$SETTA_GENERATED_PYTHON

model = model.to("cuda")

first_call = True
original_size = None


def inpaint_or_outpaint(layers, prompt, output_name):
    # global first_call, original_size
    # if first_call:
    #     first_call = False
    #     image = Image.open("input_imgs/bfl_example.jpg")
    #     original_size = image.size
    # else:
    layers = layers[1:]
    image = base64_to_pil(layers[0])
    # width, height = original_size
    width, height = image.size
    mask = layers[1]
    mask = convert_transparent_to_black(mask)
    image = model(
        **inference_args,
        prompt=prompt,
        image=image,
        mask_image=mask,
        width=width,
        height=height,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
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


def get_inpaint_output_fn(section_name, prompt_name, output_name):
    def fn(p):
        drawing = p[section_name]["drawing"]
        return inpaint_or_outpaint(drawing, p[prompt_name]["text"], output_name)
    return fn


def create_in_memory_fn(idx):
    fn = get_inpaint_output_fn(f"input_{idx}", f"prompt_{idx}", f"output_{idx}")
    return SettaInMemoryFn(fn=fn, dependencies=[f"input_{idx}['drawing']"])


# fn1 = create_in_memory_fn(1)
fn2 = create_in_memory_fn(2)
# fn3 = create_in_memory_fn(3)
# fn3 = create_in_memory_fn(4)
