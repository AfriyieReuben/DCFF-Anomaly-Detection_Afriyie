import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class GIADPreprocessor:
    def __init__(self, raw_dir, output_dir):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        
    def process_images(self):
        # Create directories
        os.makedirs(f"{self.output_dir}/train/normal", exist_ok=True)
        os.makedirs(f"{self.output_dir}/train/anomaly", exist_ok=True)
        os.makedirs(f"{self.output_dir}/test/normal", exist_ok=True)
        os.makedirs(f"{self.output_dir}/test/anomaly", exist_ok=True)
        
        # Process each image
        for img_name in os.listdir(self.raw_dir):
            img = cv2.imread(f"{self.raw_dir}/{img_name}")
            img = cv2.resize(img, (256, 256))
            
            # Split and save (example logic)
            if "normal" in img_name:
                cv2.imwrite(f"{self.output_dir}/train/normal/{img_name}", img)
            else:
                cv2.imwrite(f"{self.output_dir}/train/anomaly/{img_name}", img)

if __name__ == "__main__":
    preprocessor = GIADPreprocessor("raw_images", "processed_giad")
    preprocessor.process_images()