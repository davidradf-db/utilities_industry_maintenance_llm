# Databricks notebook source
# MAGIC %pip install transformers torch datasets databricks-vectorsearch
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

import os
# import faiss
import torch
# import skimage
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import IPython.display
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# COMMAND ----------

def get_model_info(model_ID, device):
	# Save the model to device
	model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
	processor = CLIPProcessor.from_pretrained(model_ID)
	# Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
  # Return model, processor & tokenizer
	return model, processor, tokenizer

# COMMAND ----------

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device)

# COMMAND ----------

def get_single_image_embedding(my_image,processor, model, device):
  image = processor(
      text = None,
      images = my_image,
      return_tensors="pt"
      )["pixel_values"].to(device)
  embedding = model.get_image_features(image)
  # convert the embeddings to numpy array
  return embedding.cpu().detach().numpy()

# COMMAND ----------

one_image = Image.open('./image_inspections/cropped_insulator.png')
one_vector = get_single_image_embedding(one_image, processor, model, device) # Simple test

# COMMAND ----------

df = spark.createDataFrame([{"image_embedding":one_vector.squeeze(0).tolist(),"part_no":"19-c","part_name":"insulator","replacement_part":"19-d"}])
df.write.mode("overwrite").saveAsTable(f"{catalog}.{db}.image_embedding_table")

# COMMAND ----------

one_image = Image.open('./image_inspections/cropped_worn_hook.png')
one_vector = get_single_image_embedding(one_image, processor, model, device) # Simple test

# COMMAND ----------

df = spark.createDataFrame([{"image_embedding":one_vector.squeeze(0).tolist(),"part_no":"28-alpha","part_name":"hanger hook","replacement_part":"29-charlie"}])
df.write.mode("append").saveAsTable(f"{catalog}.{db}.image_embedding_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `dbdemos`.`rag_chatbot_david_radford_newco`.`image_embedding_table` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.image_embedding_table"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.part_no_lookup"

max_attempts = 3
attempts = 0
index_created = False

while attempts < max_attempts and not index_created:
  VECTOR_SEARCH_ENDPOINT_NAME = "-".join(VECTOR_SEARCH_ENDPOINT_NAME.split("-")[:-1] + [str(attempts)])
  try:
      if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
          print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
          vsc.create_delta_sync_index(
              endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
              index_name=vs_index_fullname,
              source_table_name=source_table_fullname,
              pipeline_type="TRIGGERED", #Sync needs to be manually triggered
              primary_key="part_no",
              embedding_vector_column="image_embedding",
              embedding_dimension=512
          )
      index_created = True
  except Exception as e:
      attempts += 1
      print(f"Attempt {attempts} failed. Retrying...")
      if attempts == max_attempts:
          print(f"Max attempts reached. Index creation failed.")
          raise e

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# from databricks.vector_search.client import VectorSearchClient

# vsc = VectorSearchClient()

# index = vsc.get_index(endpoint_name="one-env-shared-endpoint-5", index_name="dbdemos.rag_chatbot_david_radford_newco.part_no_lookup")

# COMMAND ----------

# index.similarity_search(
#   query_vector=one_vector.squeeze(0).tolist(),
#   columns=['part_no']
# )

# COMMAND ----------

# one_image = Image.open('./flagged/Worn_hook_labeled.png')
# one_vector = get_single_image_embedding(one_image, processor, model, device)
# index.similarity_search(
#   query_vector=one_vector.squeeze(0).tolist(),
#   columns=['part_no']
# )

# COMMAND ----------

# import io

# def helper_to_bytes(img):
#     img_bytes = io.BytesIO()
#     img.save(img_bytes, format='PNG')
#     return img_bytes.getvalue()

# COMMAND ----------

# one_image.size

# COMMAND ----------

# import io
# _bytes = helper_to_bytes(one_image)
# # print(_bytes)
# # Image.frombytes('RGB', (128,128), _bytes, 'raw')
# Image.open(io.BytesIO(_bytes))

# # get_single_image_embedding(_bytes, processor, model, device)

# COMMAND ----------

# def get_single_image_embedding(my_image,processor, model, device):
#   image = processor(
#       text = None,
#       images = my_image,
#       return_tensors="pt"
#       )["pixel_values"].to(device)
#   embedding = model.get_image_features(image)
#   # convert the embeddings to numpy array
#   return embedding.cpu().detach().numpy()
