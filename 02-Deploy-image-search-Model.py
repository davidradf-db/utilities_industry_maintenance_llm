# Databricks notebook source
# TODO make more elegant
dbutils.widgets.text("skip_setup", "false", "Use packages in init script or skip")
skip_setup = dbutils.widgets.get("skip_setup") == "true"

if skip_setup:
  dbutils.notebook.exit("skipped")

# COMMAND ----------

# DBTITLE 1,Install the required libraries
# MAGIC %pip install --ignore-installed mlflow==2.10.2 langchain==0.1.5 databricks-vectorsearch databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false

# COMMAND ----------

# from databricks.vector_search.client import VectorSearchClient
# from langchain_community.vectorstores import DatabricksVectorSearch
# from langchain_community.embeddings import DatabricksEmbeddings

# # Test embedding Langchain model
# #NOTE: your question embedding model must match the one used in the chunk in the previous model 
# embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
# print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")

# def get_retriever(persist_dir: str = None):
#     os.environ["DATABRICKS_HOST"] = host
#     #Get the vector search index
#     vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
#     vs_index = vsc.get_index(
#         endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
#         index_name=index_name
#     )

#     # Create the retriever
#     vectorstore = DatabricksVectorSearch(
#         vs_index, text_column="content", embedding=embedding_model
#     )
#     return vectorstore.as_retriever()


# # test our retriever 
# vectorstore = get_retriever()
# similar_documents = vectorstore.get_relevant_documents("This hanger hook hole looks enlarged")
# print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

import mlflow.pyfunc
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import base64
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
class ImageSearch(mlflow.pyfunc.PythonModel):


  def get_model_info(self, model_ID, device):
    # Save the model to device
    self.model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    self.processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    self.tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
  
  def get_single_image_embedding(self, my_image,processor, model, device):
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    image = processor(
        text = None,
        images = my_image,
        return_tensors="pt"
        )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()

  def load_context(self, context):
    import os
    # import faiss
    import torch
    # import skimage
    import requests
    import numpy as np
    import pandas as pd
    from PIL import Image
    from io import BytesIO
    from datasets import load_dataset
    from collections import OrderedDict
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(personal_access_token=os.environ['DATABRICKS_TOKEN'], workspace_url=host)
    self.index = vsc.get_index(endpoint_name="one-env-shared-endpoint-0", index_name="dbdemos.rag_chatbot_david_radford_newco.part_no_lookup")
      # Set the device
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the model ID
    model_ID = "openai/clip-vit-base-patch32"
    # Get model, processor & tokenizer
    self.get_model_info(model_ID, self.device)
  
  def predict(self, context, model_input):
    import os
    # import faiss
    import torch
    # import skimage
    import requests
    import numpy as np
    import pandas as pd
    from PIL import Image
    from io import BytesIO
    from datasets import load_dataset
    from collections import OrderedDict
    from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
    from databricks.vector_search.client import VectorSearchClient
    _bytes = model_input["image_bytes"].values[0]
    print(type(_bytes))
    image_data = base64.b64decode(_bytes)
    image_bytes = BytesIO(image_data)
    _img = Image.open(image_bytes)
    _embedding = self.get_single_image_embedding(_img, self.processor,self.model,self.device)
    results = self.index.similarity_search(
      query_vector=_embedding.squeeze(0).tolist(),
      columns=['part_no','part_name','replacement_part']
    )['result']['data_array'][0]
    answer = {"part_no":results[0], "part_name":results[1], "replacement_part":results[2]}
    # answer = {"part_no":"testPart"}
    return answer

# Create a custom pyfunc model
custom_model = ImageSearch()



# COMMAND ----------

from mlflow.models import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# Option 1: Manually construct the signature object
input_schema = Schema(
    [
        ColSpec("string", "image_bytes"),
    ]
)
output_schema = Schema([ColSpec("string","part_no")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
signature

# COMMAND ----------

# Set the experiment name
# experiment_name = "/Repos/david.radford@databricks.com/utilities_industry_maintenance_llm"

# Start an MLflow run
with mlflow.start_run() as run:
  run_id = run.info.run_id
  # Log the custom model
  mlflow.pyfunc.log_model(artifact_path="model", python_model=custom_model, signature=signature,
        extra_pip_requirements=[
            "mlflow==2.10.2",
            "databricks-vectorsearch",
        ],)


# COMMAND ----------

import io
import base64
from PIL import Image
import pandas as pd
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def helper_to_bytes(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    return img_bytes.getvalue()
  
one_image = Image.open('./image_inspections/cropped_insulator.png').resize((30,30))
_ = base64.b64encode(helper_to_bytes(one_image)).decode('utf-8')
# print(_)
df = pd.DataFrame([{"image_bytes":_}])

import mlflow
logged_model = f'runs:/{run_id}/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(df)

# custom_model.load_context(None)
# custom_model.predict(None,df)

# COMMAND ----------

#Enable Unity Catalog with MLflow registry
from mlflow.tracking.client import MlflowClient
mlflow.set_registry_uri('databricks-uc')
model_name = f"{catalog}.{dbName}.image_search_model"

client = MlflowClient()
_register = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
client.set_registered_model_alias(model_name,"Production", int(_register.version))

# COMMAND ----------

f"{catalog}_{db}_image_search"

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = f"{catalog}_{db}_image_search"
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=False,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",  # <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config,timeout=3600)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    # w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

{"dataframe_records": [{"image_bytes":"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"}]}
