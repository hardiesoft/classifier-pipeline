import argparse
import os
import json

from testclassify import easy_result

"""

Script to create a json file to be used to test classify changes against

Reads metadata from downlaoded cptv videos and creates a simplified json form

Output file should be review and changed as required 

"""

DEFAULT_GROUPS = {
    0: [
        "bird",
        "false-positive",
        "hedgehog",
        "possum",
        "rodent",
        "mustelid",
        "cat",
        "kiwi",
        "dog",
        "leporidae",
        "human",
        "insect",
        "pest",
    ],
    1: ["unidentified", "other"],
    2: ["part", "bad track"],
    3: ["default"],
}


def parse_meta(folder):
    for folder_path, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1] == ".cptv":
                full_path = os.path.join(folder_path, name)
                load_meta(full_path)


def load_meta(cptv_filename):
    meta_filename = os.path.splitext(cptv_filename)[0] + ".txt"
    if not os.path.exists(meta_filename):
        return
    with open(meta_filename) as file:
        meta = json.load(file)

    recording = easy_result(meta, ai_tag=False, tag_precedence=tag_precedence)
    recording["Tracks"] = sorted(recording["Tracks"], key=lambda x: x["start_s"])
    recordings.append(recording)


def get_tag_precedence():
    tag_rec = {}
    for order, tags in DEFAULT_GROUPS.items():
        for tag in tags:
            tag_rec[tag] = order

    if tag_rec.get("default") is None:
        tag_rec["default"] = max(DEFAULT_GROUPS) + 1
    return tag_rec


recordings = []
tag_precedence = get_tag_precedence()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder container cptv files")
    parser.add_argument(
        "--output",
        dest="output_file",
        default="classify-test.json",
        help="Output json file",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    parse_meta(args.folder)
    with open(args.output_file, "w") as outfile:
        json.dump({"Recordings": recordings}, outfile, indent=2)


if __name__ == "__main__":
    main()
