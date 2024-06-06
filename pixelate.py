#!/usr/bin/python
import argparse
from pathlib import Path

import torch
import numpy as np
import huggingface_hub
import safetensors.torch as st
from PIL import Image
from tqdm import tqdm

from models.networks import define_G


def load_img(img_path, gray=False):
    img = Image.open(img_path)
    img = img.convert("L" if gray else "RGB")
    img = torch.from_numpy(np.array(img))
    img = img.to(torch.get_default_dtype()).mul_(2 / 255).sub_(1.0)
    if gray:
        img = img.expand((1, 3, -1, -1))
    else:
        img = img.permute((2, 0, 1)).unsqueeze(0).contiguous()
    return img


def to_pil(img):
    img = img[0].permute((1, 2, 0))
    img = (
        img.add(1 + 1 / 255).mul_(255 / 2).clamp_(0, 255).to(torch.uint8).cpu().numpy()
    )
    return Image.fromarray(img, mode="RGB")


def load_models(repository="gaeros/pixelization", device="cuda"):
    model_path = huggingface_hub.hf_hub_download(
        repository, filename="160_net_G_A.safetensors"
    )
    c2p = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="c2pGen",
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
        init=False,
    )
    c2p.load_state_dict(st.load_file(model_path, device=str(device)), assign=True)
    c2p.eval()

    model_path = huggingface_hub.hf_hub_download(
        repository, filename="alias_net.safetensors"
    )
    aliasnet = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="antialias",
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
        init=False,
    )
    aliasnet.load_state_dict(st.load_file(model_path, device=str(device)), assign=True)
    aliasnet.eval()

    return c2p, aliasnet


downscale_methods = {
    "linear": "bilinear",
    "cubic": "bicubic",
    "nearest": "nearest_exact",
}

if __name__ == "__main__":
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        metavar="<input>",
        type=str,
        nargs="+",
        help="input images",
    )
    parser.add_argument(
        "-r",
        "--reference",
        default=None,
        type=str,
        help="reference image for picking the pixel size",
    )
    parser.add_argument(
        "-c",
        "--cell_size",
        default=4,
        type=int,
        help="cell size",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="output folder",
    )
    parser.add_argument(
        "-s",
        "--output_suffix",
        type=str,
        default="_pixelated",
        help="output suffix",
    )
    parser.add_argument(
        "-f",
        "--output_format",
        type=str,
        default="png",
        help="output format",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=compute_device,
        help="Device to run the computations on",
    )
    parser.add_argument(
        "-p",
        "--pre_downscale",
        type=float,
        default=1.0,
        help="1.0",
    )
    parser.add_argument(
        "-m",
        "--downscale_method",
        choices=["linear", "cubic", "nearest", "none"],
        default="linear",
        help="method for downscaling the output of the network",
    )
    parser.add_argument(
        "-u",
        "--no_upscale",
        action="store_true",
        help="do not upscale to the original resolution",
    )
    args = parser.parse_args()
    device = args.device

    ref_img = load_img(
        args.reference or f"./examples/{args.cell_size}_1.png", gray=True
    )
    c2p, aliasnet = load_models(device=device)
    for input_path in tqdm(args.input):
        input_path = Path(input_path)
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        output_path = (
            output_dir / f"{input_path.stem}{args.output_suffix}.{args.output_format}"
        )

        in_img = load_img(input_path)
        if args.pre_downscale != 1.0:
            in_img = torch.nn.functional.interpolate(
                in_img, scale_factor=args.pre_downscale, mode="bicubic"
            )
        with torch.inference_mode():
            res = c2p(in_img.to(device), ref_img.to(device))
            res = aliasnet(res)

        if args.downscale_method != "none":
            cell_size = args.cell_size
            res = torch.nn.functional.interpolate(
                res,
                scale_factor=1 / cell_size,
                mode=downscale_methods[args.downscale_method],
                align_corners=False,
                antialias=False,
            )
            if not args.no_upscale:
                res = torch.nn.functional.interpolate(
                    res, scale_factor=cell_size, mode="nearest-exact"
                )

        to_pil(res).save(output_path)
