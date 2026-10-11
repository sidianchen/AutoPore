from pathlib import Path
from PIL import Image
import cv2

# ============================================================
# Main folders
# ============================================================
main_folders = [
    Path("single_run_dyn_pnm"),
    Path("single_run_qs_pnm"),
]

# ============================================================
# Crop boxes: (left, top, right, bottom)
# ============================================================
crop_box_3d = (200, 100, 1150, 1200)
crop_box_pore = (400, 100, 1120, 800)

# ============================================================
# PNG helper
# ============================================================
def crop_with_box(img, crop_box):
    width, height = img.size

    left, top, right, bottom = crop_box

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid crop box {crop_box} for image size {width} x {height}"
        )

    return img.crop((left, top, right, bottom))


# ============================================================
# AVI helper
# ============================================================
def crop_avi(video_path, output_path, crop_box):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        print(f"Empty video: {video_path}")
        cap.release()
        return

    height, width = frame.shape[:2]

    left, top, right, bottom = crop_box

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)

    crop_w = right - left
    crop_h = bottom - top

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless AVI

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (crop_w, crop_h),
    )

    while True:
        if not ret:
            break

        cropped = frame[top:bottom, left:right]
        out.write(cropped)

        ret, frame = cap.read()

    cap.release()
    out.release()

    print(f"Saved AVI: {output_path}")


def main() -> None:
    for main_folder in main_folders:
        print(f"\nProcessing folder: {main_folder}")

        output_dir = main_folder / "figs_cropped"
        output_dir.mkdir(parents=True, exist_ok=True)

        pore_network_fig = main_folder / "input_pnm" / "pore_network.png"
        image_3d_fig = main_folder / "figs" / "3D_image.png"
        pore_figs = sorted((main_folder / "figs").glob("ps_*.png"))

        png_files = list(
            dict.fromkeys([pore_network_fig, image_3d_fig] + pore_figs)
        )
        for fig_path in png_files:
            if not fig_path.exists():
                print(f"Skipped, file not found: {fig_path}")
                continue

            img = Image.open(fig_path)
            crop_box = (
                crop_box_3d
                if fig_path.name == "3D_image.png"
                else crop_box_pore
            )
            cropped = crop_with_box(img, crop_box)
            output_path = output_dir / f"{fig_path.stem}_cropped.png"
            cropped.save(output_path)
            print(f"Saved PNG: {output_path}")

        for avi_path in main_folder.rglob("*.avi"):
            crop_box = (
                crop_box_3d
                if "3d" in avi_path.stem.lower()
                else crop_box_pore
            )
            output_path = output_dir / f"{avi_path.stem}_cropped.avi"
            crop_avi(avi_path, output_path, crop_box)


if __name__ == "__main__":
    main()
