import numpy as np
import os
from PIL import Image
import argparse

def read_yuv420_frame(f, width, height):
    y_size = width * height
    uv_size = (width // 2) * (height // 2)

    y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape((height, width)).astype(np.float32)
    u = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2)).astype(np.float32)
    v = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape((height // 2, width // 2)).astype(np.float32)

    u = u.repeat(2, axis=0).repeat(2, axis=1)
    v = v.repeat(2, axis=0).repeat(2, axis=1)

    y = (y - 16.0) * (1 / 219.0)
    u = (u - 128.0) * (1 / 224.0)
    v = (v - 128.0) * (1 / 224.0)

    r = y + 1.5748 * v
    g = y - 0.1873 * u - 0.4681 * v
    b = y + 1.8556 * u

    rgb = np.stack((r, g, b), axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

def convert_yuv420_to_jpg(yuv_path, width, height, output_folder, stride):
    os.makedirs(output_folder, exist_ok=True)
    frame_size = width * height * 3 // 2
    frame_index = 0
    saved_frame_count = 0

    with open(yuv_path, 'rb') as f:
        while True:
            pos = f.tell()
            f.seek(0, 2)
            eof = f.tell()
            f.seek(pos)

            if eof - pos < frame_size:
                break

            if frame_index % stride == 0:
                rgb_frame = read_yuv420_frame(f, width, height)
                img = Image.fromarray(rgb_frame, 'RGB')
                img.save(os.path.join(output_folder, f'{saved_frame_count:04d}.jpg'), quality=95)
                saved_frame_count += 1
            else:
                f.seek(frame_size, 1)

            frame_index += 1

    print(f"Saved {saved_frame_count} JPG frames with stride {stride} to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yuv_file", type=str, help="Path to input YUV file")
    parser.add_argument("--width", type=int, default=1920, help="Frame width")
    parser.add_argument("--height", type=int, default=1080, help="Frame height")
    parser.add_argument("--output", type=str, default="frames", help="Output folder")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame (e.g., 4 means every 4th)")
    args = parser.parse_args()

    convert_yuv420_to_jpg(args.yuv_file, args.width, args.height, args.output, args.stride)
