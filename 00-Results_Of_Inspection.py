# Databricks notebook source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the flagged folder
folder_path = "./image_inspections"

# Get the path of the image to view
image_path_labeled = folder_path + "/Insulator_Labeled.png"
image_path_cropped = folder_path + "/cropped_insulator.png"

# Load and display the image
img_labeled = mpimg.imread(image_path_labeled)
plt.imshow(img_labeled)
plt.axis('off')
plt.show()

img = mpimg.imread(image_path_cropped)
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the flagged folder
folder_path = "./image_inspections"

# Get the path of the image to view
image_path_labeled = folder_path + "/Worn_hook_labeled.png"
image_path_cropped = folder_path + "/cropped_worn_hook.png"

# Load and display the image
img_labeled = mpimg.imread(image_path_labeled)
plt.imshow(img_labeled)
plt.axis('off')
plt.show()

img = mpimg.imread(image_path_cropped)
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------


