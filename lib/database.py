import numpy as np

from typing import List
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

CONNECTION_NAME = "default"
CONNECTION_HOST = "localhost"
CONNECTION_PORT = "19530"
FACE_COLLECTION = "faces"
CREATE_INDEX = True
DIST_THRESHOLD = .4


def connect_milvus() -> None:
    connections.connect(
        CONNECTION_NAME,
        host=CONNECTION_HOST,
        port=CONNECTION_PORT,
    )


def disconnect_milvus() -> None:
    connections.disconnect(CONNECTION_NAME)


def create_collection(
    name: str,
    create_index: bool,
    index_name: str,
) -> Collection:
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="identifier", dtype=DataType.INT64),
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

    collection = Collection(name, schema=schema)
    if create_index:
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 512},
        }

        collection.create_index(index_name, index_params)

    return collection


def get_collection(name: str) -> Collection:
    if not utility.has_collection(name):
        print("Creating collection...")
        return create_collection(
            name=name,
            create_index=True,
            index_name="encoding",
        )
    else:
        return Collection(name)


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
                f"pk == {id}"
                for id in ids
            ]
        ),
        output_fields=["identifier", "top", "right", "bottom", "left"],
    )

    return results