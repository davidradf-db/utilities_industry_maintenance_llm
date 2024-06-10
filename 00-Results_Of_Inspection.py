# Databricks notebook source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the flagged folder
folder_path = "./flagged"

# Get the path of the image to view
image_path = folder_path + "/Insulator_Labeled.png"

# Load and display the image
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the flagged folder
folder_path = "./flagged"

# Get the path of the image to view
image_path = folder_path + "/Worn_hook_labeled.png"

# Load and display the image
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# COMMAND ----------


