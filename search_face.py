import os
import sys
import pickle
import numpy as np
import face_recognition as fr

from PIL import Image
from typing import (
    List, 
    Dict,
    Union,
)
from encode_and_insert import FACE_COLLECTION

from lib.database import (
    connect_milvus,
    get_collection,
    search_encodings,
    query_ids,
    disconnect_milvus, 
)

from lib.face_utils import (
    get_face_features,
    get_hit_image_slice,
)

FACE_COLLECTION = "faces"


def main(input: str) -> None:
    if not os.path.isfile(input):
        print("You must provide an input image!")
        print(f"Could not find file at {input}.")
        sys.exit(0)

    image, encodings, locations = get_face_features(input)

    connect_milvus()
    collection = get_collection(FACE_COLLECTION)
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

        print(hits)
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
    disconnect_milvus()


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