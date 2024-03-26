import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_url", type=str, default="", help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--gpu_count", type=int, default=0, help="")

    parser.add_argument("--exp_name", type=int, default="meter", help="")
    parser.add_argument("--seed", type=int, default=0, help="")
    # TODO
    parser.add_argument("--datasets", type=str, default="coco,vg,sbu,gcc", help="")
    parser.add_argument("--loss_names", type=str, default={"itm": 1, "mlm": 1, "mim": 1}, help="")
    parser.add_argument("--batch_size", type=int, default=4096, help="")

    # Image setting
    parser.add_argument("--train_transform_keys", type=str, default="clip", help="")
    parser.add_argument("--val_transform_keys", type=str, default="clip", help="")
    parser.add_argument("--image_size", type=int, default=224, help="")
    parser.add_argument("--patch_size", type=int, default=32, help="")
    parser.add_argument("--draw_false_image", type=int, default=1, help="")
    parser.add_argument("--image_only", type=bool, default=False, help="")
    parser.add_argument("--resolution_before", type=int, default=224, help="")

    # Text Setting
    parser.add_argument("--vqav2_label_size", type=int, default=3129, help="")
    parser.add_argument("--max_text_len", type=int, default=40, help="")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="")
    parser.add_argument("--vocab_size", type=int, default=30522, help="")
    parser.add_argument("--whole_word_masking", type=bool, default=False, help="")
    parser.add_argument("--mlm_prob", type=int, default=0.15, help="")
    parser.add_argument("--draw_false_text", type=int, default=0, help="")

    # Transformer Setting
    parser.add_argument("--num_top_layer", type=int, default=6, help="")
    parser.add_argument("--input_image_embed_size", type=int, default=768, help="")
    parser.add_argument("--input_text_embed_size", type=int, default=768, help="")
    parser.add_argument("--vit", type=str, default='ViT-B-32.pt', help="")
    parser.add_argument("--hidden_size", type=int, default=768, help="")
    parser.add_argument("--num_heads", type=int, default=12, help="")
    parser.add_argument("--num_layers", type=int, default=6, help="")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="")
    parser.add_argument("--drop_rate", type=int, default=0.1, help="")

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = ""
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    return parser
