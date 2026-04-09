#!/usr/bin/env python3
"""Generate near-duplicate image dataset for testing the deduplicator."""

import argparse
import os
import sys
import random

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("Error: requires Pillow and numpy. Install with: pip3 install Pillow numpy")
    sys.exit(1)


def jpeg_recompress(img, quality):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf)
    result.load()
    return result


def add_noise(img, intensity=10):
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def center_crop(img, fraction=0.95):
    w, h = img.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def rotate_small(img, angle):
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))


def random_crop(img, fraction=0.80):
    w, h = img.size
    new_w, new_h = int(w * fraction), int(h * fraction)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h))


def color_shift(img):
    arr = np.array(img, dtype=np.float32)
    shifts = np.random.uniform(-15, 15, size=3)
    arr += shifts[np.newaxis, np.newaxis, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def overlay_text(img):
    from PIL import ImageDraw
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    draw.rectangle([w - 120, h - 30, w, h], fill=(255, 255, 255, 180))
    draw.text((w - 115, h - 25), "sample", fill=(180, 180, 180))
    return overlay


TRANSFORMS = [
    ("jpeg_q70", lambda img: jpeg_recompress(img, 70)),
    ("jpeg_q50", lambda img: jpeg_recompress(img, 50)),
    ("jpeg_q30", lambda img: jpeg_recompress(img, 30)),
    ("flip_h", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    ("flip_v", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ("rot5", lambda img: rotate_small(img, 5)),
    ("rot15", lambda img: rotate_small(img, 15)),
    ("rot90", lambda img: img.transpose(Image.ROTATE_90)),
    ("rot180", lambda img: img.transpose(Image.ROTATE_180)),
    ("crop95", lambda img: center_crop(img, 0.95)),
    ("crop85", lambda img: center_crop(img, 0.85)),
    ("crop_rand80", lambda img: random_crop(img, 0.80)),
    ("bright_up", lambda img: ImageEnhance.Brightness(img).enhance(1.25)),
    ("bright_down", lambda img: ImageEnhance.Brightness(img).enhance(0.75)),
    ("contrast_up", lambda img: ImageEnhance.Contrast(img).enhance(1.3)),
    ("saturate", lambda img: ImageEnhance.Color(img).enhance(1.4)),
    ("desaturate", lambda img: ImageEnhance.Color(img).enhance(0.5)),
    ("color_shift", lambda img: color_shift(img)),
    ("noise", lambda img: add_noise(img, 15)),
    ("heavy_noise", lambda img: add_noise(img, 30)),
    ("blur", lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5))),
    ("sharpen", lambda img: img.filter(ImageFilter.SHARPEN)),
    ("downscale", lambda img: img.resize((img.width // 2, img.height // 2), Image.LANCZOS)),
    ("upscale_blur", lambda img: img.resize((img.width // 2, img.height // 2), Image.LANCZOS).resize((img.width, img.height), Image.LANCZOS)),
    ("jpeg_flip", lambda img: jpeg_recompress(img.transpose(Image.FLIP_LEFT_RIGHT), 60)),
    ("crop_bright", lambda img: ImageEnhance.Brightness(center_crop(img, 0.90)).enhance(1.15)),
    ("blur_noise", lambda img: add_noise(img.filter(ImageFilter.GaussianBlur(radius=1)), 10)),
    ("watermark", lambda img: overlay_text(img)),
]


def generate_variants(img_path, output_dir, num_variants):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"  Warning: cannot open {img_path}: {e}")
        return 0

    basename = os.path.splitext(os.path.basename(img_path))[0]

    out_path = os.path.join(output_dir, f"{basename}_original.png")
    img.save(out_path)

    selected = random.sample(TRANSFORMS, min(num_variants, len(TRANSFORMS)))
    count = 1

    for name, transform_fn in selected:
        try:
            variant = transform_fn(img)
            out_path = os.path.join(output_dir, f"{basename}_{name}.png")
            variant.save(out_path)
            count += 1
        except Exception as e:
            print(f"  Warning: transform '{name}' failed on {basename}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Generate near-duplicate image dataset")
    parser.add_argument("input_dir", help="Directory containing base images")
    parser.add_argument("output_dir", help="Directory to write generated dataset")
    parser.add_argument("--variants", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.input_dir):
        print(f"Error: input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tga", ".tiff"}
    image_files = []
    for fname in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(fname)[1].lower() in supported_ext:
            image_files.append(os.path.join(args.input_dir, fname))

    if not image_files:
        print(f"Error: no images found in '{args.input_dir}'")
        sys.exit(1)

    print(f"Found {len(image_files)} base images")
    print(f"Generating {args.variants} variants each -> ~{len(image_files) * (args.variants + 1)} total images")
    print(f"Output directory: {args.output_dir}")
    print()

    total = 0
    for i, path in enumerate(image_files):
        count = generate_variants(path, args.output_dir, args.variants)
        total += count
        print(f"  [{i+1}/{len(image_files)}] {os.path.basename(path)} -> {count} images")

    print(f"\nDone! Generated {total} images in {args.output_dir}/")


if __name__ == "__main__":
    main()
