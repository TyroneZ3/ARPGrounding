import argparse

import torch
from tqdm import tqdm

from datasets.arpgrounding import get_dataset
from inference_grounding import interpret_albef, interpret_blip, interpret_clip, meter_make_batch, interpret_meter

try:
    import CLIP.clip as clip
except Exception:
    pass

try:
    from lavis.models import load_model_and_preprocess
except Exception:
    pass

try:
    from meter.config import config as meter_config
    from meter.datamodules.datamodule_base import get_pretrained_tokenizer
    from meter.modules import METERTransformerSS
    from meter.transforms import keys_to_transforms as meter_keys_to_transforms
except Exception:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-nW", "--nW", default=0, help="number of workers", required=False)
    parser.add_argument("-Isize", "--Isize", default=304, help="image size", required=False)
    parser.add_argument("-clip_eval", "--clip_eval", default=False, action="store_true", help="", required=False)
    parser.add_argument("-meter_eval", "--meter_eval", default=False, action="store_true", help="", required=False)
    parser.add_argument("-albef_eval", "--albef_eval", default=False, action="store_true", help="", required=False)
    parser.add_argument("-blip2_eval", "--blip2_eval", default=False, action="store_true", help="", required=False)
    parser.add_argument("-clip_path", "--clip_path", type=str, default="ViT-B/32", help="clip path or name", required=False)
    parser.add_argument("-albef_path", "--albef_path", type=str, default="", help="albef folder path", required=False)
    parser.add_argument("-blip2_path", "--blip2_path", type=str, default="", help="blip2 folder path", required=False)
    parser.add_argument("-val_path", "--val_path", default="../../wsg/MultiGrounding/data/visual_genome", help="data set path", required=False)
    parser.add_argument("-split", "--split", default="attribute", help="composition type: attribute/relation/priority", required=True)
    args = vars(parser.parse_args())

    # Load dataset
    ds = get_dataset(args)
    ds.files = list(ds.annotations.keys())
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=int(args["nW"]), shuffle=False, drop_last=False)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args["clip_eval"]:
        clip_model, _ = clip.load(args["clip_path"], device=device, jit=False)
    elif args["meter_eval"]:
        config = meter_config()
        meter_model = METERTransformerSS(config).to(device)
        transform = meter_keys_to_transforms(config["val_transform_keys"], size=config["image_size"])[0]
        ds.transform = transform
        meter_tokenizer = get_pretrained_tokenizer(config["tokenizer"])
        meter_max_text_len = config["max_text_len"]
    elif args["albef_eval"]:
        albef_model, _, _ = load_model_and_preprocess("albef_image_text_matching", "coco", device=device, is_eval=True)
        if args["albef_path"]:
            checkpoint = torch.load(args["albef_path"], map_location="cpu")
            state_dict = checkpoint
            albef_model.load_state_dict(state_dict, strict=False)
    elif args["blip2_eval"] or args["path_ae"]:
        blip_model, _, _ = load_model_and_preprocess("blip2_image_text_matching", "coco", device=device, is_eval=True)
        if args["blip2_path"]:
            checkpoint = torch.load(args["blip2_path"], map_location="cpu")
            state_dict = checkpoint["state_dict"]
            blip_model.load_state_dict(state_dict)

    # Inference
    pbar = tqdm(dl)
    cnt_overall1, cnt_overall2, cnt_overall = 0, 0, 0
    cnt_correct1, cnt_correct2, cnt_correct = 0, 0, 0
    pos_area, neg_area = 0, 0
    for i, inputs in enumerate(pbar):
        real_imgs, meta, size, _ = inputs
        if len(list(meta.keys())) == 0:
            continue
        real_imgs = real_imgs.cuda()
        real_imgs = real_imgs.repeat(len(meta), 1, 1, 1)
        size = [int(size[0]), int(size[1])]
        texts, neg_texts, pos_bboxes, neg_bboxes = [], [], [], []
        for item in list(meta.values()):
            texts.append(item["sentences"][0])
            neg_texts.append(item["neg_sentences"][0])
            pos_bboxes.append(
                [
                    int(item["bbox"][0][0] / size[1] * real_imgs.size(3)),
                    int(item["bbox"][0][1] / size[0] * real_imgs.size(2)),
                    int(item["bbox"][0][2] / size[1] * real_imgs.size(3)),
                    int(item["bbox"][0][3] / size[0] * real_imgs.size(2)),
                ]
            )
            neg_bboxes.append(
                [
                    int(item["bbox"][1][0] / size[1] * real_imgs.size(3)),
                    int(item["bbox"][1][1] / size[0] * real_imgs.size(2)),
                    int(item["bbox"][1][2] / size[1] * real_imgs.size(3)),
                    int(item["bbox"][1][3] / size[0] * real_imgs.size(2)),
                ]
            )
        # for pos_bbox, neg_bbox in zip(pos_bboxes, neg_bboxes):
        #     pos_area += (pos_bbox[2] - pos_bbox[0]) * (pos_bbox[3] - pos_bbox[1])
        #     neg_area += (neg_bbox[2] - neg_bbox[0]) * (neg_bbox[3] - neg_bbox[1])
        # prnt = pos_area / (pos_area + neg_area)
        # pbar.set_description(str(prnt))
        # continue

        try:
            if args["clip_eval"]:  # use clip
                assert real_imgs.size()[2:] == (224, 224)
                tokenized_text = clip.tokenize(texts).to("cuda")
                neg_tokenized_text = clip.tokenize(neg_texts).to("cuda")
                heatmaps = interpret_clip(real_imgs, tokenized_text, clip_model, device=device).detach()
                neg_heatmaps = interpret_clip(real_imgs, neg_tokenized_text, clip_model, device=device).detach()

            elif args["meter_eval"]:
                assert real_imgs.size()[2:] == (384, 384)
                for p in meter_model.parameters():
                    p.requires_grad = False
                batch = meter_make_batch(real_imgs[0], texts, meter_tokenizer, meter_max_text_len)
                neg_batch = meter_make_batch(real_imgs[0], neg_texts, meter_tokenizer, meter_max_text_len)
                heatmaps = interpret_meter(batch, meter_model, "cuda")
                neg_heatmaps = interpret_meter(neg_batch, meter_model, "cuda")

            elif args["albef_eval"]:
                assert real_imgs.size()[2:] == (384, 384)
                heatmaps = interpret_albef(real_imgs, texts, albef_model, device=device).detach()
                neg_heatmaps = interpret_albef(real_imgs, neg_texts, albef_model, device=device).detach()

            elif args["blip2_eval"]:
                assert real_imgs.size()[2:] == (364, 364)
                heatmaps = interpret_blip(real_imgs, texts, blip_model, device=device).detach()
                neg_heatmaps = interpret_blip(real_imgs, neg_texts, blip_model, device=device).detach()

            for j in range(0, len(heatmaps)):
                pos_regions = [
                    heatmaps[j, 0, pos_bboxes[j][1] : pos_bboxes[j][3], pos_bboxes[j][0] : pos_bboxes[j][2]],
                    neg_heatmaps[j, 0, neg_bboxes[j][1] : neg_bboxes[j][3], neg_bboxes[j][0] : neg_bboxes[j][2]]
                ]
                neg_regions = [
                    heatmaps[j, 0, neg_bboxes[j][1] : neg_bboxes[j][3], neg_bboxes[j][0] : neg_bboxes[j][2]],
                    neg_heatmaps[j, 0, pos_bboxes[j][1] : pos_bboxes[j][3], pos_bboxes[j][0] : pos_bboxes[j][2]]
                ]

                if (
                    pos_regions[0].size(0) * pos_regions[0].size(1) * neg_regions[0].size(0) * neg_regions[0].size(1) == 0
                    or pos_regions[1].size(0) * pos_regions[1].size(1) * neg_regions[1].size(0) * neg_regions[1].size(1) == 0
                ):  # skip invalid bboxes
                    continue

                # pos_act = pos_region.mean()
                # neg_act = neg_region.mean()
                pos_act = [pos_region.mean() for pos_region in pos_regions]
                neg_act = [neg_region.mean() for neg_region in neg_regions]
                if pos_act[0] > neg_act[0]:
                    cnt_correct1 += 1
                cnt_overall1 += 1
                if pos_act[1] > neg_act[1]:
                    cnt_correct2 += 1
                cnt_overall2 += 1
                if pos_act[0] > neg_act[0] and pos_act[1] > neg_act[1]:
                    cnt_correct += 1
                cnt_overall += 1
            correctness1 = 100.0 * cnt_correct1 / cnt_overall1
            correctness2 = 100.0 * cnt_correct2 / cnt_overall2
            correctness = 100.0 * cnt_correct / cnt_overall
            prnt = "correctness1:{:.2f}%, correctness2:{:.2f}%, correctness:{:.2f}%".format(correctness1, correctness2, correctness)
            pbar.set_description(prnt)

        except Exception as e:  # skip OOM samples
            print(str(e))

    print(prnt)
