import os
import pickle
import torch
from PIL import Image

from datasets.tfs import get_flicker_transform


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None, loader=pil_loader):
        annotations_path_dict = {
            "priority": "arpgrounding/priority_vg.pkl",
            "attribute": "arpgrounding/attribute_vg.pkl",
            "relation": "arpgrounding/relationship_vg.pkl",
        }
        self.annotations = pickle.load(open(annotations_path_dict[split], "rb"))
        self.img_folder = os.path.join(root, "VG_Images")

        self.split = split
        self.transform = transform
        self.loader = loader
        self.files = list(self.annotations.keys())
        print("num of data:{}".format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        ann = self.annotations[int(item)]  # [[region, object], ...]

        img_path = os.path.join(self.img_folder, item + ".jpg")
        img = pil_loader(img_path)
        image_sizes = (img.height, img.width)
        img = self.transform(img)

        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            tmp["sentences"] = ann[i][0]["phrase"].lower()
            tmp["neg_sentences"] = ann[i][1]["phrase"].lower()
            tmp["bbox"] = [
                [
                    int(ann[i][0]["x"]),
                    int(ann[i][0]["y"]),
                    int(ann[i][0]["x"]) + int(ann[i][0]["w"]),
                    int(ann[i][0]["y"]) + int(ann[i][0]["h"]),
                ],
                [
                    int(ann[i][1]["x"]),
                    int(ann[i][1]["y"]),
                    int(ann[i][1]["x"]) + int(ann[i][1]["w"]),
                    int(ann[i][1]["y"]) + int(ann[i][1]["h"]),
                ]
            ]

            out[str(i)] = tmp

        return img, out, image_sizes, img_path

    def __len__(self):
        return len(self.files) * 1


def get_dataset(args):
    datadir = args["val_path"]
    split = args["split"]
    transform_train, transform_test = get_flicker_transform(args)
    ds_test = ImageLoader(datadir, split, transform=transform_test)
    return ds_test
