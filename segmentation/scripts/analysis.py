import pandas as pd
import matplotlib.pyplot as plt

# Correct path to segmentation results.csv
csv_path = '/home/sumeet/model_training/segmentation/scripts/runs/train/ground_segmentation5/results.csv'

# Load the CSV file
df = pd.read_csv(csv_path)

# Clean column names
df.columns = df.columns.str.strip()

# Print available columns
print("Available columns:", df.columns.tolist())

# Define segmentation-specific metrics
columns_to_plot = [
    'metrics/precision(M)',
    'metrics/recall(M)',
    'metrics/mAP50(M)',
    'metrics/mAP50-95(M)'
]

# Plotting
plt.figure(figsize=(10, 6))
for col in columns_to_plot:
    if col in df.columns:
        plt.plot(df['epoch'].values, df[col].values, label=col)
    else:
        print(f"Column '{col}' not found in CSV!")

plt.title('Ground Segmentation Performance Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("segmentation_metrics.png")
plt.show()
