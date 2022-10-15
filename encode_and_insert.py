import os

from lib.database import (
    connect_milvus,
    get_collection,
    disconnect_milvus,
)

from lib.face_utils import (
    dump_state,
    encode_images,
    create_entities,
)


ROOT_IMAGE_PATH = "images"
FACE_COLLECTION = "faces"


def reset_all() -> None:
    global ID2IMG, IMG2ID

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
        collection = get_collection(FACE_COLLECTION)
        collection.drop()
    except:
        pass

    ID2IMG = {}
    IMG2ID = {}
    dump_state()


def main() -> None:
    connect_milvus()
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
    disconnect_milvus()

if __name__ == "__main__":
    main()