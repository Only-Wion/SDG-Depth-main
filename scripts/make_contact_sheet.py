import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="outputs/kitti_batch_10/batch_metrics.json")
    parser.add_argument("--output", default="outputs/kitti_batch_10/contact_sheet.png")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thumb_width = 600
    thumb_height = 338
    label_height = 28
    columns = 2
    rows = (len(report["samples"]) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)

    for position, sample in enumerate(report["samples"]):
        image_path = report_path.parent / Path(sample["visualization"]).name
        if not image_path.exists():
            image_path = Path(sample["visualization"])
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        x = (position % columns) * thumb_width
        y = (position // columns) * (thumb_height + label_height)
        canvas.paste(image, (x + (thumb_width - image.width) // 2, y))
        metrics = sample["metrics"]
        draw.text(
            (x + 8, y + thumb_height + 6),
            f"Sample {sample['index']}: RMSE {metrics['rmse_m']:.3f} m | MAE {metrics['mae_m']:.3f} m",
            fill="black",
        )

    canvas.save(output_path, quality=95)
    print(output_path)


if __name__ == "__main__":
    main()
