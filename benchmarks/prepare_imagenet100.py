#!/usr/bin/env python3
"""Prepare ImageNet-100 from full ImageNet-1k or download a pre-selected subset.

ImageNet-100 uses the 100 classes selected by Tian et al. (CMC, 2020),
which has become the standard subset for SSL benchmarks.

Usage:
    # If you have ImageNet-1k already:
    python -m benchmarks.prepare_imagenet100 --imagenet-dir /path/to/imagenet --output-dir ./data/imagenet100

    # If you don't have ImageNet and want to use a HuggingFace mirror:
    python -m benchmarks.prepare_imagenet100 --from-huggingface --output-dir ./data/imagenet100
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# The 100 ImageNet class IDs from Tian et al. (CMC, ECCV 2020).
# These are the standard 100 classes used across SSL papers.
IMAGENET100_CLASSES = [
    "n02869837", "n01749939", "n02488291", "n02107142", "n13037406",
    "n02091831", "n04517823", "n04589890", "n03062245", "n01773797",
    "n01735189", "n07831146", "n07753275", "n03085013", "n04485082",
    "n02105505", "n01983481", "n02788148", "n03530642", "n04435653",
    "n02086910", "n02859443", "n13040303", "n03594734", "n02085620",
    "n02099849", "n01558993", "n04493381", "n02109047", "n04111531",
    "n02877765", "n04429376", "n02009229", "n01978455", "n02106550",
    "n01820546", "n01692333", "n07714571", "n02974003", "n02114855",
    "n03384352", "n02088466", "n02091244", "n02701002", "n02165456",
    "n02669723", "n03637318", "n01980166", "n02236044", "n03452741",
    "n02422699", "n02510455", "n02966193", "n04040759", "n02483362",
    "n04347754", "n04596742", "n04019541", "n02105855", "n04398044",
    "n02093859", "n02125311", "n04153751", "n02206856", "n03793489",
    "n02814533", "n03017168", "n02219486", "n02487347", "n02104365",
    "n02002556", "n01776313", "n02077923", "n04067472", "n02190166",
    "n04389033", "n03599486", "n02071294", "n02018207", "n02950826",
    "n03075370", "n01770393", "n03891332", "n02708093", "n04265275",
    "n04275548", "n01774750", "n02087394", "n02489166", "n02364673",
    "n04131690", "n03400231", "n02281406", "n01531178", "n02113799",
    "n03272010", "n02138441", "n03544143", "n03761084", "n04604644",
]


def create_from_imagenet(imagenet_dir, output_dir):
    """Create ImageNet-100 by symlinking from full ImageNet-1k."""
    imagenet_dir = Path(imagenet_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        src = imagenet_dir / split
        if not src.exists():
            print(f"WARNING: {src} not found, skipping {split} split")
            continue

        dst = output_dir / split
        dst.mkdir(parents=True, exist_ok=True)

        found = 0
        for cls_id in IMAGENET100_CLASSES:
            cls_src = src / cls_id
            cls_dst = dst / cls_id
            if cls_src.exists():
                if not cls_dst.exists():
                    os.symlink(cls_src.resolve(), cls_dst)
                found += 1

        n_images = sum(len(list((dst / c).iterdir())) for c in os.listdir(dst) if (dst / c).is_dir())
        print(f"{split}: {found}/100 classes, {n_images} images -> {dst}")

    print(f"\nImageNet-100 ready at {output_dir}")


def create_from_huggingface(output_dir):
    """Download ImageNet-100 using HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  # required for HuggingFace download")
        sys.exit(1)

    output_dir = Path(output_dir)

    print("Downloading ImageNet-100 from HuggingFace...")
    print("This may take a while depending on your connection.")

    # Use the standard ImageNet-100 subset on HuggingFace
    ds = load_dataset("imagenet-1k", split="train",
                      trust_remote_code=True, streaming=False)

    # Filter to our 100 classes
    # Map synset IDs to integer labels
    cls_to_idx = {cls_id: i for i, cls_id in enumerate(IMAGENET100_CLASSES)}

    for split in ["train", "validation"]:
        out_split = "val" if split == "validation" else split
        print(f"Processing {split}...")

        ds_split = load_dataset("imagenet-1k", split=split, trust_remote_code=True)

        count = 0
        for item in ds_split:
            label = item["label"]
            # This depends on the HF dataset format — may need class name mapping
            synset = ds_split.features["label"].int2str(label)
            if synset in cls_to_idx:
                cls_dir = output_dir / out_split / synset
                cls_dir.mkdir(parents=True, exist_ok=True)
                img = item["image"]
                img.save(cls_dir / f"{count:06d}.JPEG")
                count += 1

        print(f"  Saved {count} images for {split}")

    print(f"\nImageNet-100 ready at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-100")
    parser.add_argument("--imagenet-dir", type=str, default=None,
                        help="Path to full ImageNet-1k (ILSVRC2012)")
    parser.add_argument("--from-huggingface", action="store_true",
                        help="Download from HuggingFace")
    parser.add_argument("--output-dir", type=str, default="./data/imagenet100",
                        help="Output directory")
    args = parser.parse_args()

    if args.imagenet_dir:
        create_from_imagenet(args.imagenet_dir, args.output_dir)
    elif args.from_huggingface:
        create_from_huggingface(args.output_dir)
    else:
        print("Provide either --imagenet-dir or --from-huggingface")
        print("\nIf you have ImageNet-1k already:")
        print("  python -m benchmarks.prepare_imagenet100 --imagenet-dir /path/to/ILSVRC2012")
        print("\nOtherwise:")
        print("  python -m benchmarks.prepare_imagenet100 --from-huggingface")
        sys.exit(1)


if __name__ == "__main__":
    main()
