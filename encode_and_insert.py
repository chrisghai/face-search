import os
import pickle
import platform
import numpy as np
import face_recognition as fr

from distutils import util
from PIL import Image
from typing import (
    List,
    Dict,
    Tuple,
)

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

if 'arm64' in platform.machine() and 'mac' in util.get_platform():
    os.environ["GRPC_PYTHON_BUILD_SYSTEM_OPENSSL"] = "1"
    os.environ["GRPC_PYTHON_BUILD_SYSTEM_ZLIB"] = "1"
else:
    pass

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

CONNECTION_NAME = "encode-insert"
CONNECTION_HOST = "localhost"
CONNECTION_PORT = "19530"
FACE_COLLECTION = "faces"
CREATE_INDEX = True
ROOT_IMAGE_PATH = "images/"
FACE_SHAPE_THRESHOLD = 64

def create_collection(
    name: str,
    create_index: bool,
    index_name: str,
) -> Collection:
    fields = [
        FieldSchema(name="identifier", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="encoding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="top", dtype=DataType.INT64),
        FieldSchema(name="right", dtype=DataType.INT64),
        FieldSchema(name="bottom", dtype=DataType.INT64),
        FieldSchema(name="left", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Collection of face encodings and positional information.",
    )

    collection = Collection(name, schema=schema, using=CONNECTION_NAME)
    if create_index:
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 512},
        }

        collection.create_index(index_name, index_params)

    return collection


def get_collection(name: str) -> Collection:
    if not utility.has_collection(name, using=CONNECTION_NAME):
        print("Creating collection...")
        return create_collection(
            name=name,
            create_index=True,
            index_name="encoding",
        )
    else:
        return Collection(name, using=CONNECTION_NAME)


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

        image_path = os.path.join(root, image)
        image_dot_path = os.path.join(root, "." + image)

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
            data[image_dot_path] = (image_id, encodings, locations)
            IMG2ID[image_dot_path] = image_id
            ID2IMG[image_id] = image_dot_path

    return data


def show_target(
    image_reference: str, 
    location: List[int],
) -> bool:
    img = fr.load_image_file(image_reference)
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


def dump_state() -> None:
    with open(img2id_path, "wb") as f:
        pickle.dump(IMG2ID, f)

    with open(id2img_path, "wb") as f:
        pickle.dump(ID2IMG, f)


def reset_all() -> None:
    global IMG2ID, ID2IMG

    images = os.listdir(ROOT_IMAGE_PATH)
    for image in images:
        if "DS_Store" in image:
            continue

        if image.startswith("."):
            os.rename(
                os.path.join(ROOT_IMAGE_PATH, image),
                os.path.join(ROOT_IMAGE_PATH, image[1:]),
            )

    try:
        collection = Collection(FACE_COLLECTION)
        collection.drop()
    except:
        pass

    ID2IMG = {}
    IMG2ID = {}
    dump_state()


def main() -> None:
    connections.connect(
        CONNECTION_NAME,
        host=CONNECTION_HOST,
        port=CONNECTION_PORT,
    )
    print("Connected.")

    reset = ""
    while (
        not reset or
        reset not in ["y", "n"]
    ):
        reset = input("Reset images, collection and state? y/n: ")

    if reset == "y":
        print("Resetting...", end="")
        reset_all()
        print(" done!")

    print("Getting collection...")
    face_collection = get_collection(FACE_COLLECTION)

    print("Encoding images...")
    face_data = encode_images(ROOT_IMAGE_PATH)

    print("Creating entities...")
    db_entities = create_entities(face_data)

    if db_entities:
        print("Inserting entitites...", sep="")
        face_collection.insert(db_entities)
        print(" done!")

    dump_state()
    connections.disconnect(CONNECTION_NAME)

if __name__ == "__main__":
    main()