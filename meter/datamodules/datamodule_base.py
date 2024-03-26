import torch

# from pytorch_lightning import LightningDataModule
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if 'roberta' in from_pretrained:
                RobertaTokenizer.from_pretrained(from_pretrained)
            else:
                BertTokenizer.from_pretrained(
                    from_pretrained, do_lower_case="uncased" in from_pretrained
                )
        torch.distributed.barrier()

    if 'roberta' in from_pretrained:
        return RobertaTokenizer.from_pretrained(from_pretrained)
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )
