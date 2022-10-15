import os
import sys
import pickle
import numpy as np
import face_recognition as fr

from PIL import Image
from typing import (
    List,
    Dict,
    Tuple,
    Union,
)

os.makedirs("state", exist_ok=True)
img2id_path = "state/img2id.pkl"
id2img_path = "state/id2img.pkl"

if os.path.isfile(img2id_path):
    with open(img2id_path, "rb") as f:
        IMG2ID = pickle.load(f)
else:
    IMG2ID = {}

if os.path.isfile(id2img_path):
    with open(id2img_path, "rb") as f:
        ID2IMG = pickle.load(f)
else:
    ID2IMG = {}

FACE_SHAPE_THRESHOLD = 64
IMAGE_ROOT_FOLDER = "images"


def dump_state() -> None:
    with open(img2id_path, "wb") as f:
        pickle.dump(IMG2ID, f)

    with open(id2img_path, "wb") as f:
        pickle.dump(ID2IMG, f)


def encode_images(
    root: str = "images/",
) -> Dict[str, Tuple]:
    data = {}
    images = os.listdir(root)
    if not images:
        print(f"Found no images in {root}!")
        print(
            "Either place images there,",
            "change ROOT_IMAGE_PATH or reset state.",
        )

    for image in images:
        if image.startswith("."):
            continue

        if "DS_Store" in image:
            continue

        image_dot = "." + image
        image_path = os.path.join(root, image)
        image_dot_path = os.path.join(root, image_dot)

        print(f"Getting features of {image_path}...")
        im = fr.load_image_file(image_path)
        locations = fr.face_locations(
            im,
            number_of_times_to_upsample=2,
        )

        encodings = fr.face_encodings(
            im, 
            known_face_locations=locations,
            num_jitters=3, 
            model="small",
        )

        os.rename(
            image_path, 
            image_dot_path,
        )

        if len(encodings) > 0:
            image_id = sum(
                [
                    ord(c)
                    for c in image
                ]
            )
            data[image_dot] = (image_id, encodings, locations)
            IMG2ID[image_dot] = image_id
            ID2IMG[image_id] = image_dot

    return data


def show_target(
    image_reference: str, 
    location: List[int],
) -> bool:
    img = fr.load_image_file(
        os.path.join(IMAGE_ROOT_FOLDER, image_reference),
    )

    img = img[
        location[0]:location[2],
        location[3]:location[1],
    ]

    if not all(np.array(img.shape[:-1]) >= FACE_SHAPE_THRESHOLD):
        return False

    Image.fromarray(img).show()
    return True


def create_entities(
    data: Dict[str, Tuple],
) -> List[list]:
    if not data:
        return None

    ids, enc = [], []
    top, right, bottom, left = [], [], [], []
    for image_reference, (image_id, encodings, locations) in data.items():
        for encoding, location in zip(encodings, locations):
            if not show_target(image_reference, location):
                print("User face size is too small. Skipping...")
                continue

            choice = ""
            while (
                not choice or
                choice.lower()[0] not in ["y", "n"]
            ):
                choice = input("Is this a user? y/n: ")

            if choice == "n":
                continue

            ids.append(image_id)
            enc.append(encoding)

            t, r, b, l = location
            top.append(t)
            right.append(r)
            bottom.append(b)
            left.append(l)

    return [ids, enc, top, right, bottom, left]


def get_face_features(
    input: str,
) -> Tuple[np.ndarray, list, list]:
    img = fr.load_image_file(input)
    _locations = fr.face_locations(
        img,
        number_of_times_to_upsample=2,
    )

    encodings = fr.face_encodings(
        img,
        known_face_locations=_locations,
        num_jitters=3,
    )

    if len(encodings) == 0:
        print("Found no faces in input!")
        sys.exit(0)

    locations = []
    for location in _locations:
        locations.append(
            {
                "top": location[0],
                "right": location[1],
                "bottom": location[2],
                "left": location[3],
            }
        )

    return img, encodings, locations


def get_image(identifier: int) -> Union[None, np.ndarray]:
    img_reference = ID2IMG.get(identifier)
    if not img_reference:
        return None

    return fr.load_image_file(
        os.path.join(IMAGE_ROOT_FOLDER, img_reference),
    )


def get_hit_image_slice(
    query_result: Dict[str, int],
) -> Union[None, np.ndarray]:
    image = get_image(query_result["identifier"])
    if image is None:
        return None

    image_slice = image[
        query_result["top"]:query_result["bottom"],
        query_result["left"]:query_result["right"],
    ]

    return Image.fromarray(image_slice)