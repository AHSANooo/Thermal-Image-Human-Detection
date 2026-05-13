import argparse

import cv2

from pipeline import pipeline


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction and posture classification for one image.")
    parser.add_argument("image_path", help="Path to an input image (jpg or png).")
    parser.add_argument(
        "--save-overlay",
        default="posture_overlay.png",
        help="Output path for overlay image (default: posture_overlay.png).",
    )
    args = parser.parse_args()

    result, valid_humans, features, postures = pipeline(args.image_path)

    if not valid_humans:
        print("No valid humans detected in the image.")
        return

    for idx, (human, feat, posture) in enumerate(zip(valid_humans, features, postures), start=1):
        print(f"Human {idx}:")
        print(f"  bbox   : {human['bbox']}")
        print(f"  area   : {human['area']}")
        if human.get("center") is not None:
            print(f"  center : {human['center']}")
        print(f"  features: {feat}")
        print(f"  posture : {posture}")

    # Overlay bounding boxes and posture labels on the original image
    original_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    for idx, (human, posture) in enumerate(zip(valid_humans, postures), start=1):
        x, y, w, h = map(int, human["bbox"])
        label = f"Human{idx}: {posture}"
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(args.save_overlay, overlay)
    print(f"Overlay saved to {args.save_overlay}")


if __name__ == "__main__":
    main()
