"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.albef import AlbefBase
from lavis.models.base_model import MomentumDistilationMixin, SharedQueueMixin
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn


@registry.register_model("albef_image_text_matching")
class AlbefITM(AlbefBase, MomentumDistilationMixin, SharedQueueMixin):
    """
    ALBEF retrieval model.

    Supported model types:
        - coco: fine-tuned ALBEF base model on COCO dataset (Karparthy split).
        - flickr: fine-tuned ALBEF base model on Flickr30k dataset.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("albef_retrieval", "coco")
        >>> model = load_model("albef_retrieval", "flickr")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "coco": "configs/albef_retrieval_coco.yaml",
        "flickr": "configs/albef_retrieval_flickr.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        queue_size,
        embed_dim=256,
        temp=0.07,
        use_distill=True,
        momentum=0.995,
        alpha=0.4,
        max_txt_len=30,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

        # create the momentum encoder
        self.visual_encoder_m = deepcopy(self.visual_encoder)
        self.text_encoder_m = deepcopy(self.text_encoder)

        self.vision_proj_m = deepcopy(self.vision_proj)
        self.text_proj_m = deepcopy(self.text_proj)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.text_encoder, self.text_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(temp * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len
        self.use_distill = use_distill

    def forward(self, samples, mode="itm"):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). The input images.
                - text_input (list): A list of length batch_size, each element is a string of text/caption.
            mode (str): The mode of the model. Supported modes are "itm" and "itc".

        Returns:
        """
        image = samples["image"]
        caption = samples["text_input"]

        image_embeds = self.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.text_encoder.forward_text(text)
        text_embeds = text_output.last_hidden_state

        if mode == "itc":
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

            sims = image_feat @ text_feat.t()
            sim = sims.diag()

            return sim

        elif mode == "itm":
            encoder_output_pos = self.text_encoder(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode="fusion",
            )

            vl_embeddings = encoder_output_pos.last_hidden_state[:, 0, :]
            itm_logits = self.itm_head(vl_embeddings)

            return itm_logits


    @classmethod
    def from_config(cls, cfg=None):
        image_encoder = VisionTransformerEncoder.from_config(cfg, from_pretrained=False)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        momentum = cfg.get("momentum", 0.995)
        alpha = cfg.get("alpha", 0.4)
        temp = cfg.get("temp", 0.07)
        max_txt_len = cfg.get("max_txt_len", 30)
        queue_size = cfg.get("queue_size", 0)
        use_distill = cfg.get("use_distill", True)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            queue_size=queue_size,
            embed_dim=embed_dim,
            temp=temp,
            momentum=momentum,
            alpha=alpha,
            max_txt_len=max_txt_len,
            use_distill=use_distill,
        )

        model.load_checkpoint_from_config(cfg)

        return model
