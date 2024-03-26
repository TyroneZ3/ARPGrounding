import os
import pickle
import json
import random
import torch
import numpy as np
from multiprocessing import Pool
from PIL import Image
from datasets.tfs import get_flicker_transform
# import spacy
from tqdm import tqdm


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ImageLoader(torch.utils.data.Dataset):

    collate_fn = None

    def __init__(self, root, transform=None, split="train", loader=pil_loader):

        self.imgs_data_folder = os.path.join(root, "VG_Annotations", "imgs_data.pickle")
        self.splits_folder = os.path.join(root, "VG_Splits", "data_splits.pickle")
        self.annotations_folder = os.path.join(root, "VG_Annotations", "region_descriptions.json")
        with open(self.annotations_folder, "rb") as f:
            self.annotations = json.load(f, encoding="latin1")
        with open(self.splits_folder, "rb") as f:
            self.splits = pickle.load(f, encoding="latin1")
        with open(self.imgs_data_folder, "rb") as f:
            self.imgs_data = pickle.load(f, encoding="latin1")
        self.img_folder = os.path.join(root, "VG_Images")

        self.transform = transform
        self.loader = loader
        self.files = list(self.splits[split])
        self.split = split
        self.annotations = sync_data(self.files, self.annotations, self.imgs_data, split)
        self.files = list(self.annotations.keys())
        print("num of data:{}".format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + ".jpg")
        img = pil_loader(img_path)
        image_sizes = (img.height, img.width)
        img = self.transform(img)
        ann = self.annotations[int(item)]
        if self.split == "train":
            region_id = np.random.randint(0, len(ann))
            return img, ann[region_id]["phrase"].lower()
        out = {}
        for i in range(0, len(ann)):
            tmp = {}
            tmp["sentences"] = ann[i]["phrase"].lower()
            bbox = [
                [
                    int(ann[i]["x"]),
                    int(ann[i]["y"]),
                    int(ann[i]["x"]) + int(ann[i]["width"]),
                    int(ann[i]["y"]) + int(ann[i]["height"]),
                ]
            ]
            tmp["bbox"] = bbox
            if (bbox[0][3] - bbox[0][1]) * (bbox[0][2] - bbox[0][0]) > 0.05 * image_sizes[0] * image_sizes[1]:
                out[str(i)] = tmp
        return img, out, image_sizes, img_path

    def __len__(self):
        return len(self.files) * 1


class BucketImageLoader(ImageLoader):
    def __init__(self, data_root, transform=None, split="train", loader=pil_loader):
        assert split == "train", "BucketImageLoader only support training"
        self.imgs_data_folder = os.path.join(data_root, "VG_Annotations", "imgs_data.pickle")
        self.splits_folder = os.path.join(data_root, "VG_Splits", "data_splits.pickle")
        with open(self.splits_folder, 'rb') as f:
            self.splits = pickle.load(f, encoding='latin1')
        with open(self.imgs_data_folder, 'rb') as f:
            self.imgs_data = pickle.load(f, encoding='latin1')
        self.img_folder = os.path.join(data_root, "VG_Images")

        self.transform = transform
        self.loader = loader
        self.files = list(self.splits[split])
        self.split = split
        # Get annotations
        if not os.path.exists(os.path.join(data_root, 'bucket_text.pickle')):
            self.parser = spacy.load('en_core_web_sm')
            self.annotations_folder = os.path.join(data_root, "VG_Annotations", "region_descriptions.json")
            with open(self.annotations_folder, 'rb') as f:
                self.annotations = json.load(f, encoding='latin1')
            self.annotations = sync_data(self.files, self.annotations, self.imgs_data, split)
            # Get bucket annotations
            self.num_proc = 10
            # Slice annotations for multiprocessing
            listed_annotations = list(self.annotations.items())
            sliced_annotations = [dict(listed_annotations[i::self.num_proc]) for i in range(self.num_proc)]
            self.annotations = {}
            with Pool(self.num_proc) as pool:
                bucket_annotations = pool.map(self.get_bucket_annotations, sliced_annotations)
                for bucket_annotation in bucket_annotations:
                    self.annotations.update(bucket_annotation)
            # Save bucket annotations
            with open(os.path.join(data_root, 'bucket_text.pickle'), 'wb') as f:
                pickle.dump(self.annotations, f)
        else:
            self.annotations = pickle.load(open(os.path.join(data_root, 'bucket_text.pickle'), 'rb'))
        self.files = list(self.annotations.keys())
        self.attributes = {
            'white': 'black', 'black': 'white', 'green': 'yellow', 'blue': 'red', 'red': 'blue', 'brown': 'yellow',
            'yellow': 'green', 'gray': 'white', 'orange': 'blue', 'purple': 'yellow', 'pink': 'blue', 'large': 'small',
            'small': 'large', 'closed': 'open', 'open': 'closed', 'sitting': 'standing', 'standing': 'sitting',
            'tall': 'short', 'short': 'tall', 'long': 'short', 'wet': 'dry', 'dry': 'wet', 'leafy': 'leafless',
            'leafless': 'leafy'}
        print('num of data:{}'.format(len(self.files)))

    def __getitem__(self, index):
        item = str(self.files[index])
        img_path = os.path.join(self.img_folder, item + '.jpg')
        img = pil_loader(img_path)
        image_sizes = (img.height, img.width)
        img = self.transform(img)
        ann = self.annotations[int(item)]
        if self.split == 'train':
            buckets = ann
            # Sample bucket
            buckets = list(buckets.values())
            sentences = random.choices(buckets, weights=(len(b) for b in buckets))[0]
            # Sample sentences
            if len(sentences) > 1:
                sentences = random.sample(sentences, 2)
            else:
                sentences = sentences * 2
            # Get negative sentences
            neg_sentences = []
            # for i, sent in enumerate(sentences):
            #     neg_sentence = self.get_neg_text(sent)
            #     if neg_sentence:
            #         neg_sentences.append(neg_sentence)
            return img, sentences + neg_sentences

    def get_root(self, text):
        doc = self.parser(text)
        sent_root = None
        for token in doc:
            if token.dep_ == 'ROOT':
                if 'NN' in token.tag_:
                    return token.lemma_.lower()
                else:
                    for child in token.children:
                        if 'NN' in child.tag_:
                            return child.lemma_.lower()
            # if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
            #     return token.lemma_.lower()
            # elif token.dep_ == 'ROOT':
            #     sent_root = token
        if sent_root:
            return sent_root.lemma_.lower()
        else:
            return sent_root

    def get_bucket_annotations(self, annotations):
        keys = annotations.keys()
        for key in tqdm(keys):
            bucket = {}
            for ann in annotations[key]:
                sent_root = self.get_root(ann['phrase'])
                if sent_root is None:  # if can't get root, print and skip
                    # print(f"Annotation key: {key}, Sentence: {ann['phrase'].lower()}, Root: {sent_root}")
                    pass
                elif sent_root in bucket:
                    bucket[sent_root].append(ann['phrase'].lower())
                else:
                    bucket[sent_root] = [ann['phrase'].lower()]
            annotations[key] = bucket
        return annotations

    def get_neg_text(self, sentence):
        tokens = sentence.split(' ')
        is_modified = False
        for token in tokens:
            if token in self.attributes:
                sentence = sentence.replace(token, self.attributes[token])
                is_modified = True
        if not is_modified:
            return None
        return sentence

    @classmethod
    def collate_fn(cls, batch):
        imgs = []
        targets = []
        neg_imgs = []
        neg_targets = []
        for sample in batch:
            imgs += [sample[0]] * 2  # repeat image
            targets += sample[1][:2]
            neg_imgs += [sample[0]] * len(sample[1][2:])
            neg_targets += sample[1][2:]
        return torch.stack(imgs + neg_imgs, 0), targets + neg_targets


def get_VG_dataset(args):
    datadir = args["data_path"]
    transform_train, transform_test = get_flicker_transform(args)
    if not args["w4"] and not args["w5"]:
        ds_train = ImageLoader(datadir, split="train", transform=transform_train)
    else:
        ds_train = BucketImageLoader(datadir, split="train", transform=transform_train)
    return ds_train


def get_VGtest_dataset(args):
    datadir = args["val_path"]
    transform_train, transform_test = get_flicker_transform(args)
    ds_test = ImageLoader(datadir, split="test", transform=transform_test)
    return ds_test


def sync_data(files, annotations, imgs_data, split="train"):
    out = {}
    for ann in tqdm(annotations):
        if ann["id"] in files:
            tmp = []
            for item in ann["regions"]:
                if len(item["phrase"].split(" ")) < 80:
                    tmp.append(item)
            out[ann["id"]] = tmp
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-Isize", "--Isize", default=224, help="image size", required=False)
    args = vars(parser.parse_args())
    ds = get_VG_dataset(args=args)
    ds = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, drop_last=False)
    pbar = tqdm(ds)
    for i, (real_imgs, text) in enumerate(pbar):
        pass
    # for i, (real_imgs, meta, size, img_path) in enumerate(pbar):
    #     size = [int(size[1]), int(size[0])]
    #     image = real_imgs.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    #     image = (image - image.min()) / (image.max() - image.min())
    #     image = np.array(255*image).copy().astype(np.uint8)
    #     image = cv2.resize(image, size)
    #     for sen in meta:
    #         item = sen['phrase']
    #         bbox = int(sen['x']), int(sen['y']), int(sen['x']) + int(sen['width']), int(sen['y']) + int(sen['height'])
    #         (gxa, gya, gxb, gyb) = bbox
    #         image = cv2.rectangle(image, (gxa, gya), (gxb, gyb), (0, 0, 255), 2)
    #         cv2.imwrite('kaki.jpg', image)
