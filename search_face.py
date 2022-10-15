import os
import sys
import pickle
import platform
import numpy as np
import face_recognition as fr

from distutils import util
from PIL import Image
from typing import (
    List, 
    Dict,
    Union,
    Tuple,
)

from pymilvus import (
    connections,
    Collection,
)

if 'arm64' in platform.machine() and 'mac' in util.get_platform():
    os.environ["GRPC_PYTHON_BUILD_SYSTEM_OPENSSL"] = "1"
    os.environ["GRPC_PYTHON_BUILD_SYSTEM_ZLIB"] = "1"
else:
    pass

CONNECTION_NAME = "search-faces"
CONNECTION_HOST = "localhost"
CONNECTION_PORT = "19530"
FACE_COLLECTION = "faces"
IMAGE_DB_ROOT = "images"
DIST_THRESHOLD = 0.4

os.makedirs("state", exist_ok=True)
try:
    with open("state/id2img.pkl", "rb") as f:
        ID2IMG = pickle.load(f)
except:
    print("Found no id2img.pkl file!")
    print(
        "You probably need to run encode_and_insert.py first",
        "or reset state and re-run encode_and_insert.py."
    )
    sys.exit(0)


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


def search_encodings(
    collection: Collection,
    encodings: List[np.ndarray],
) -> List[list]:
    results = collection.search(
        data=encodings,
        anns_field="encoding",
        param={
            "metric_type": "L2",
            "params": {
                "nprobe": 128,
            }
        },
        limit=5,
    )

    results = [
        [
            hit for hit in result
            if hit.distance < DIST_THRESHOLD
        ] 
        for result in results
    ]

    return results


def query_ids(
    collection: Collection,
    ids: List[int],
) -> List[dict]:
    results = collection.query(
        expr=" or ".join(
            [
                f"identifier == {id}"
                for id in ids
            ]
        ),
        output_fields=["top", "right", "bottom", "left"],
    )

    return results


def get_image(identifier: int) -> Union[None, np.ndarray]:
    img_reference = ID2IMG.get(identifier)
    if not img_reference:
        return None

    return fr.load_image_file(img_reference)


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


def main(input: str) -> None:
    if not os.path.isfile(input):
        print("You must provide an input image!")
        print(f"Could not find file at {input}.")
        sys.exit(0)

    image, encodings, locations = get_face_features(input)

    connections.connect(
        CONNECTION_NAME,
        host=CONNECTION_HOST,
        port=CONNECTION_PORT,
    )

    collection = Collection(
        FACE_COLLECTION, 
        using=CONNECTION_NAME,
    )
    if collection.num_entities == 0:
        print("Collection is empty!")
        print("You need to run encode_and_insert.py script first!")
        sys.exit(0)

    collection.load()
    results = search_encodings(collection, encodings)
    for i, (location, hits) in enumerate(zip(locations, results)):
        ids = [h.id for h in hits]
        if not ids:
            print("No potential matches for this face.")
            continue

        query_results = query_ids(collection, ids)
        tmp_image = image[
            location["top"]:location["bottom"],
            location["left"]:location["right"],
        ]

        print(f"Input image location {i}:")
        Image.fromarray(tmp_image).show()
        for j, qr in enumerate(query_results, start=1):
            print(f"\tMatch {j}:")
            match_img = get_hit_image_slice(qr)
            if match_img is not None:
                match_img.show()

    print("No more matches.")
    connections.disconnect(CONNECTION_HOST)


if __name__ == "__main__":
    if (
        len(sys.argv) == 1
        or len(sys.argv) > 2
    ):
        print("Usage: python search_face.py <path-to-image>")
        sys.exit(0)

    inputfile = sys.argv[-1]
    if not os.path.isfile(inputfile):
        print(f"Found no file at {inputfile}.")
        sys.exit(0)

    main(inputfile)