import cv2
import os
import glob
import pandas as pd

from pipeline import pipeline


def batch_process_for_areas(input_dir):


    """Process all images and collect silhouette areas"""

    results = []
    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg"))

    print(f"Found {len(image_files)} images")

    # DIAGNOSTIC: List all unique files
    print("\nFiles in dataset:")
    for f in image_files:
        print(f"  - {os.path.basename(f)}")

    # Check for duplicates (same file content)
    import hashlib
    file_hashes = {}
    for f in image_files:
        with open(f, 'rb') as file:
            file_hash = hashlib.md5(file.read()).hexdigest()
            if file_hash in file_hashes:
                print(f"  ⚠️ WARNING: {os.path.basename(f)} is identical to {file_hashes[file_hash]}")
            else:
                file_hashes[file_hash] = os.path.basename(f)

    print()  # empty line



    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)

        # Your existing pipeline
        processed, humans = pipeline(img_path)

        # Collect data
        for i, human in enumerate(humans):
            x, y, w, h = human[0], human[1], human[2], human[3]
            area = human[4]

            results.append({
                'filename': filename,
                'person_id': i,
                'bbox_w': w,
                'bbox_h': h,
                'silhouette_area': area,
                'fill_ratio': area / (w * h) if (w * h) > 0 else 0
            })

            print(f"  Person {i}: area={area}px, bbox={w}×{h}")

        # Show progress every 10 images
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    if len(df) > 0:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total persons detected: {len(df)}")
        print(f"Min silhouette area: {df['silhouette_area'].min()}")
        print(f"Max silhouette area: {df['silhouette_area'].max()}")
        print(f"Avg silhouette area: {df['silhouette_area'].mean():.0f}")
        print(f"Std deviation: {df['silhouette_area'].std():.0f}")

        # Suggested thresholds (mean ± 2 std deviations)
        min_threshold = max(0, df['silhouette_area'].mean() - 2 * df['silhouette_area'].std())
        max_threshold = df['silhouette_area'].mean() + 2 * df['silhouette_area'].std()
        print(f"\nSuggested min area: {min_threshold:.0f}")
        print(f"Suggested max area: {max_threshold:.0f}")

        # Save to CSV
        df.to_csv("batch_results.csv", index=False)
        print("\nResults saved to batch_results.csv")
    else:
        print("No humans detected in any images!")

    return df

# Run it
results_df = batch_process_for_areas("dataset/")