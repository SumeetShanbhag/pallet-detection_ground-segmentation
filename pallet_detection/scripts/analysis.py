import pandas as pd
import matplotlib.pyplot as plt

# Path to your detection results.csv
csv_path = '/home/sumeet/model_training/pallet_detection/runs/train/pallet_detection4/results.csv'

# Load the CSV file
df = pd.read_csv(csv_path)

# Clean column names
df.columns = df.columns.str.strip()

# Print columns for verification (optional)
print("Available columns:", df.columns.tolist())

# Define detection-specific metrics (for bounding boxes)
columns_to_plot = [
    'metrics/precision(B)',
    'metrics/recall(B)',
    'metrics/mAP50(B)',
    'metrics/mAP50-95(B)'
]

# Plotting
plt.figure(figsize=(10, 6))
for col in columns_to_plot:
    if col in df.columns:
        plt.plot(df['epoch'].values, df[col].values, label=col)
    else:
        print(f"Column '{col}' not found in CSV!")

plt.title('Pallet Detection Performance Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("detection_metrics.png")
plt.show()
