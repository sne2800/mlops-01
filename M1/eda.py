import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sweetviz as sv
import matplotlib.pyplot as plt
import seaborn as sns

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Convert to DataFrame for analysis
df_train = pd.DataFrame(train_images.reshape(train_images.shape[0], -1))
df_train['label'] = train_labels

# Ensure the directory exists before writing
OUTPUT_DIR = "/workspaces/mlops-01/M1/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate Sweetviz EDA report
report = sv.analyze(df_train, pairwise_analysis="off")

# Save EDA report
report_path = os.path.join(OUTPUT_DIR, "eda_report.html")
report.show_html(report_path, open_browser=False)
# Run `python3 -m http.server 8080` to visualize the report

# Save Summary statistics
summary_stats_path = os.path.join(OUTPUT_DIR,"summary_statistics.csv")
df_train.describe().to_csv(summary_stats_path)

# Generate class distribution plot
plt.figure(figsize=(8, 4))
sns.countplot(x=df_train['label'], hue=df_train['label'], palette="viridis", legend=False)
plt.title("Class Distribution in Fashion MNIST")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.grid(axis="y")

# Save class distribution plot
class_dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
plt.savefig(class_dist_path)
plt.close()

# Show sample images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(train_images[i], cmap="gray")
    ax.axis("off")
plt.suptitle("Sample Images from Fashion MNIST")

# Save and display sample images plot
sample_images_path = os.path.join(OUTPUT_DIR,"sample_images.png")
plt.savefig(sample_images_path)

print(f"\nReport saved at: {report_path}")
print(f"Summary stats saved at: {summary_stats_path}")
print(f"Class Distribution Plot saved at: {class_dist_path}")
print(f"Sample images from Fashion MNIST dataset saved at: {sample_images_path}")
