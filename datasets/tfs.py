from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# from datasets.randaugment import RandomAugment
# from utils import *


resizedict = {'224': 256,
              '448': 512,
              '112': 128}


def get_aircraft_transform(size=224):
    return get_cub_transform(size=size)


def get_car_transform(size=224):
    return get_cub_transform(size=size)


def get_dog_transform(size=224):
    return get_cub_transform(size=size)


def get_flowers_transform(size=224):
    from PIL import Image
    transform_train, transform_test = get_cub_transform(size=size)
    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test, transform_mask


def get_cub_seg_transform(size=224):
    from PIL import Image
    transform_train, transform_test = get_cub_transform(size=size)
    transform_mask = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test, transform_mask


def get_nabirds_transform(size=224):
    return get_cub_transform(size=size)

def get_tiny_transform():
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transform_train = transforms.Compose([transforms.Resize((72, 72)), transforms.RandomCrop((64, 64))] + [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        normalize
    ])

    return transform_train, transform_test


def get_imagenet_transform():
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    transform_train = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop((224, 224))] + [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    return transform_train, transform_test


def get_cub_transform(size=224):
    if size == 448:
        resize = (512, 512)
        cropsize = (448, 448)
    else:
        resize = (256, 256)
        cropsize = (224, 224)
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    if resize == cropsize:
        tflist = [transforms.RandomResizedCrop(cropsize)]
    else:
        tflist = [transforms.Resize(resize),
                  transforms.RandomCrop(cropsize)]

    transform_train = transforms.Compose(tflist + [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    transform_test = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.CenterCrop(cropsize),
                             transforms.ToTensor(),
                             normalize
                             ])

    return transform_train, transform_test


def get_flicker_transform(args):
    Isize = int(args['Isize'])
    # normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    resize = (Isize, Isize)
    # cropsize = (224, 224)

    transform_train = transforms.Compose([
        transforms.Resize(resize),
        # transforms.RandomResizedCrop(
        #     Isize,
        #     scale=(0.5, 1.0),
        #     interpolation=InterpolationMode.BICUBIC,
        # ),
        transforms.RandomHorizontalFlip(),
        # RandomAugment(
        #     2,
        #     5,
        #     isPIL=True,
        #     augs=[
        #         "Identity",
        #         "AutoContrast",
        #         "Brightness",
        #         "Sharpness",
        #         "Equalize",
        #         "ShearX",
        #         "ShearY",
        #         "TranslateX",
        #         "TranslateY",
        #         "Rotate",
        #     ],
        # ),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        normalize,
    ])

    return transform_train, transform_test
