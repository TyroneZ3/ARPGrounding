import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
import textwrap


def intensity_to_rgb(intensity, cmap="cubehelix", normalize=False):
    """
    Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
    This function requires matplotlib. See `matplotlib colormaps
    <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
    list of available colormap.
    Args:
        intensity (np.ndarray): array of intensities such as saliency.
        cmap (str): name of the colormap to use.
        normalize (bool): if True, will normalize the intensity so that it has
            minimum 0 and maximum 1.
    Returns:
        np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
    """
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype("float32") * 255.0


def save_image(image, heatmaps, texts=None):
    """
    image: 3xHxW
    heatmaps: Nx1xHxW
    texts: [str]
    """
    # Interpolate image and heatmaps to 224x224
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    heatmaps = F.interpolate(heatmaps, size=(224, 224), mode="bilinear", align_corners=False)

    # Preprocess image
    mean = (
        torch.tensor((0.48145466, 0.4578275, 0.40821073))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(3, image.size(1), image.size(2))
    )
    std = (
        torch.tensor((0.26862954, 0.26130258, 0.27577711))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(3, image.size(1), image.size(2))
    )
    # mean = torch.tensor((0.5, 0.5, 0.5)).unsqueeze(-1).unsqueeze(-1).expand(3, image.size(1), image.size(2))
    # std = torch.tensor((0.5, 0.5, 0.5)).unsqueeze(-1).unsqueeze(-1).expand(3, image.size(1), image.size(2))
    image = image.cpu() * std + mean
    pil_img = ToPILImage()(image)

    # Prepare text
    if not texts:
        texts = [""] * heatmaps.size(0)
    font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, size=12)
    char_width = 8
    char_height = 18
    scale = 1.3
    texts = [text[9:] for text in texts]
    line_num = [len(textwrap.wrap(text, width=scale * image.size(2) / char_width)) for text in texts]
    max_line_num = max(line_num)

    # Prepare output image
    out = Image.new(
        "RGB", ((1 + heatmaps.size(0)) * image.size(2), image.size(1) + char_height * max_line_num), "white"
    )
    out.paste(pil_img, (0, 0))
    draw_out = ImageDraw.Draw(out)

    for i in range(heatmaps.size(0)):
        heatmap = heatmaps[i].squeeze().cpu().numpy()
        heatmap = intensity_to_rgb(heatmap)
        heatmap = heatmap.astype(np.uint8)
        pil_heatmap = Image.fromarray(heatmap)
        pil_blend_img = Image.blend(pil_img, pil_heatmap, 0.8)
        out.paste(pil_blend_img, ((i + 1) * image.size(2), 0))

        # Add text to output image
        for j, line in enumerate(textwrap.wrap(texts[i], width=scale * image.size(2) / char_width)):
            draw_out.text(((i + 1) * image.size(2), image.size(1) + char_height * j), line, font=font, fill=(0, 0, 0))

    out.save("image.jpg")


class TrainLogger:
    def __init__(self, args):
        results_path = os.path.join(args["log_path_root"], "gpu" + args["folder"], "results.csv")
        best_path = os.path.join(args["log_path_root"], "gpu" + args["folder"], "best.csv")
        self.f_all = open(results_path, "a")
        self.f_best = open(best_path, "a")
        # Header
        header_str = ["epoch", 
                      "  acc",
                      "  loss",
                      " regul" if args["w0"] else None,
                      "   cam" if args["w1"] else None,
                      "    bg" if args["w2"] else None,
                      "    fr" if args["w3"] else None,
                      "consis" if args["w4"] else None,
                      "   neg" if args["w5"] else None]
        header_str = ",".join([s for s in header_str if s is not None]) + "\n"
        if not args["resume"]:
            self.f_all.write(header_str)
            self.f_best.write(header_str)

        self.loss_dict = {}
        self.best = 0
        self.args = args

    def record(self, loss_dict):
        for k, v in loss_dict.items():
            if v is None:
                continue
            if k not in self.loss_dict:
                self.loss_dict[k] = [v]
            else:
                self.loss_dict[k].append(v)

    def log(self, acc, epoch, model):
        loss_str = [f"{np.mean(v):.4f}" for v in self.loss_dict.values()]
        loss_str = ",".join(loss_str)
        self.f_all.write(f"{epoch:5d},{acc:5.2f},{loss_str},\n")
        self.f_all.flush()

        if acc > self.best:
            torch.save(model, self.args["path_best"])
            self.best = acc
            self.f_best.write(f"{epoch:5d},{acc:.2f},{loss_str},\n")
            self.f_best.flush()

        self.loss_dict = {}


def interpret(image, text, model, device, index=None):
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    if index is None:
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    one_hot.backward(retain_graph=False)

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    for i, blk in enumerate(image_attn_blocks):
        if i <= 10:
            continue
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.detach().reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode="bilinear", align_corners=False)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    del image_attn_blocks, R, one_hot, grad, cam
    torch.cuda.empty_cache()
    return image_relevance


def interpret_clip(images, texts, model, device):
    bs = images.shape[0]
    batch_size = texts.shape[0]

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    layer = image_attn_blocks[-1]
    logits_per_image, logits_per_text = model(images, texts)
    one_hot = logits_per_image.diag().sum()
    model.zero_grad()
    grad = torch.autograd.grad(one_hot, [layer.attn_probs], retain_graph=True)[0].detach()
    cam = layer.attn_probs.detach()
    grad = grad.clamp(min=0)
    cam = grad * cam
    cam = cam.reshape(bs, -1, cam.shape[-1], cam.shape[-1])[:, :, 0, 1:].mean(1)  # B,HW

    dim = int(cam.shape[1] ** 0.5)
    cam = cam.reshape(bs, 1, dim, dim)

    image_relevance = cam.detach().clone()
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode="bicubic", align_corners=False)

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


def meter_compute_itm(pl_module, batch):
    itm_labels = torch.ones(len(batch["text_ids"])).to(pl_module.device)

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret


def interpret_meter(batch, model, device):
    bs = batch["image"][0].shape[0]

    layer = model.cross_modal_image_layers[-3].attention.self
    for p in layer.parameters():
        p.requires_grad = True
    layer.save_attention = True
    itm_dict = meter_compute_itm(model, batch)
    layer.save_attention = False
    itm_loss = itm_dict["itm_loss"]
    model.zero_grad()
    grad = torch.autograd.grad(-itm_loss, [layer.attention_map], retain_graph=True)[0].detach()
    cam = layer.attention_map.detach()
    grad = grad.clamp(min=0)

    cam = grad * cam
    cam = cam[:, :, 0, 1:].mean(1)

    dim = int(cam.shape[1] ** 0.5)
    cam = cam.reshape(bs, 1, dim, dim)

    image_relevance = cam.detach().clone()
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=384, mode="bicubic", align_corners=False)

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


def interpret_albef(images, texts, model, device):
    bs = images.shape[0]

    layer_i = -4
    layer = model.text_encoder.encoder.layer[layer_i].crossattention.self
    layer.save_attention = True
    itc_score = model({"image": images, "text_input": texts}, mode="itm")
    itc_score = torch.mean(itc_score[:, 1])
    attention_map = layer.get_attention_map()
    layer.save_attention = False

    model.zero_grad()
    cam = attention_map.detach()
    grad = torch.autograd.grad(itc_score, [attention_map], retain_graph=True)[0].detach()
    grad = grad.clamp(min=0)
    tokenized_text = model.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors='pt'
    ).to(device)
    token_mask = tokenized_text.attention_mask.view(bs, 1, -1, 1)
    token_length = tokenized_text.attention_mask.sum(dim=-1) - 2

    cam = cam * token_mask
    grad = grad * token_mask

    cam = grad * cam
    cam = cam[:, :, :, 1:].sum(dim=2).mean(dim=1) / (token_length + 2).unsqueeze(-1)

    image_relevance = cam.detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=384, mode="bilinear", align_corners=False)

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance


def interpret_blip(images, texts, model, device):
    bs = images.shape[0]

    if model.__class__.__name__ == "BlipITM":
        layer_i = -4
        layer = model.text_encoder.base_model.base_model.encoder.layer[layer_i].crossattention.self 
    elif model.__class__.__name__ == "Blip2ITM":
        layer_i = -6
        layer = model.Qformer.bert.encoder.layer[layer_i].crossattention.self
    layer.save_attention = True
    itc_score = model({'image': images, 'text_input': texts}, match_head='itm')
    itc_score = torch.mean(itc_score[:, 1])
    attention_map = layer.get_attention_map()
    layer.save_attention = False

    model.zero_grad()
    cam = attention_map.detach()
    grad = torch.autograd.grad(itc_score, [attention_map], retain_graph=True)[0].detach()
    grad = grad.clamp(min=0)

    cam = grad * cam
    cam = cam[:, :, :, 1:].mean(2).mean(1)  # B,head,Q,HW+1 -> B,HW

    image_relevance = cam.detach().clone()

    dim = int(image_relevance.shape[1] ** 0.5)
    image_relevance = image_relevance.reshape(bs, 1, dim, dim)
    if model.__class__.__name__ == "BlipITM":
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=384, mode="bilinear", align_corners=False)
    elif model.__class__.__name__ == "Blip2ITM":
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=364, mode="bicubic", align_corners=False)

    min_value = image_relevance.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    max_value = image_relevance.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).expand(image_relevance.size())
    image_relevance = (image_relevance - min_value) / (max_value - min_value)
    return image_relevance
