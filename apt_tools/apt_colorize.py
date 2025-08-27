import numpy as np
from PIL import Image
import argparse
import os
import matplotlib.pyplot as plt


class APTColorizer2D:
    def __init__(self, lut_path="palettes/noaa-apt-daylight.png"):
        self.lut = self._load_lut(lut_path)

    def _load_lut(self, path):
        lut_img = Image.open(path).convert("RGB")
        lut_arr = np.array(lut_img)
        if lut_arr.shape[0] != 256 or lut_arr.shape[1] != 256:
            lut_arr = np.array(Image.fromarray(lut_arr).resize((256, 256), Image.Resampling.BILINEAR))
        return lut_arr

    def colorize(self, data, outfile=None, show=False):
        if isinstance(data, str):
            img = Image.open(data).convert("L")
            gray_array = np.array(img)
        elif isinstance(data, np.ndarray):
            gray_array = data
        else:
            raise TypeError("Data must be a filepath or a NumPy 2D array")

        if gray_array.ndim != 2:
            raise ValueError("Input must be 2D grayscale image")

        if gray_array.shape[0] % 2 != 0:
            gray_array = gray_array[:-1]

        ch_a = gray_array[::2].astype(np.int16)
        ch_b = gray_array[1::2].astype(np.int16)

        brightness = np.clip(((ch_a + ch_b) / 2), 0, 255).astype(np.uint8)
        delta = np.clip((ch_a - ch_b + 128), 0, 255).astype(np.uint8)

        colored_half = self.lut[delta, brightness]
        colored_full = np.repeat(colored_half, 2, axis=0)

        if outfile is not None:
            Image.fromarray(colored_full).save(outfile)
            print(f"[OK] Saved to {outfile}")
        if show:
            image = Image.fromarray(colored_full)
            image.show() 

        return colored_full


def main():
    parser = argparse.ArgumentParser(description="APT Image Colorizer with 2D LUT")
    parser.add_argument("input", help="Grayscale image path")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--lut", default="palettes/noaa-apt-daylight.png", help="Path to 256x256 LUT image")
    parser.add_argument("--show", action="store_true", help="Display image after processing")
    args = parser.parse_args()

    colorizer = APTColorizer2D(args.lut)
    colorizer.colorize(args.input, outfile=args.output, show=args.show)


if __name__ == "__main__":
    main()
