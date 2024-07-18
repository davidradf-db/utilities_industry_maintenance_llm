# Databricks notebook source
# MAGIC %pip install --upgrade gradio==3.38.0 fastapi==0.104 uvicorn==0.24
# MAGIC # %pip install --upgrade gradio==4.5 fastapi==0.104 uvicorn==0.24
# MAGIC %pip install typing-extensions==4.8.0 --upgrade
# MAGIC %pip install -q -U langchain==0.0.319
# MAGIC %pip install --force-reinstall databricks-genai-inference==0.1.1
# MAGIC %pip install --ignore-installed mlflow==2.10.2 langchain==0.1.5 databricks-vectorsearch databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false $use_old_langchain=true

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
    self.index = vsc.get_index(endpoint_name="one-env-shared-endpoint-0", index_name=f"{catalog}.{db}.part_no_lookup")
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
custom_model.load_context(None)



# COMMAND ----------

# Test Databricks Foundation LLM model
from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)
print(f"Test chat model: {chat_model.predict('What is Apache Spark')}")

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is Apache Spark?')[:20]}...")
index_name=f"{catalog}.{db}.repair_reports_self_managed_vs_index"
def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    return vectorstore.as_retriever()


# test our retriever 
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents("This hanger hook hole looks enlarged")
# print(f"Relevant documents: {similar_documents[0]}")


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """You are a helpful aid for electrical lineman to help them identify repairs and how to fix thme.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = {"query": "The insulator looks like it has track marks"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# MAGIC %md ### Helper functions

# COMMAND ----------

from langchain.llms import Databricks

chatbot_model_serving_endpoint = f"{catalog}_{db}_image_search"
workspaceUrl = spark.conf.get("spark.databricks.workspaceUrl")


# def transform_input(**request):
#   full_prompt = f"""{request["prompt"]}
#   Explain in bullet points."""
#   request["query"] = full_prompt
#   # request["stop"] = ["."]
#   return request


def transform_input(**request):
  
  request["image_bytes"] = request["prompt"]
  return request


def transform_output(response):
  # Extract the answer from the responses.
  part_no = response['part_no']
  part_name = response['part_name']
  replacement_part = response['replacement_part']
  return f"Original Part Number: {part_no}\nPart Name: {part_name}\nReplacement Part Number: {replacement_part}"


# This model serving endpoint is created in `02_mlflow_logging_inference`
# llm = Databricks(host=workspaceUrl, endpoint_name=chatbot_model_serving_endpoint,
#                  transform_input_fn=transform_input, transform_output_fn=transform_output)

# COMMAND ----------

def generate_output(img: str):
    
    output = custom_model.predict(None, pd.DataFrame([{"image_bytes":img}]))

    # output = llm.invoke(img)
    return transform_output(output)

# COMMAND ----------

# MAGIC %md
# MAGIC # Maintence Question Bot

# COMMAND ----------

from langchain.llms import Databricks

maint_chatbot_model_serving_endpoint = f"{catalog}_{db}"
workspaceUrl = spark.conf.get("spark.databricks.workspaceUrl")


def question_transform_input(**request):
  full_prompt = f"""{request["prompt"]}
  Explain in bullet points."""
  request["query"] = full_prompt
  # request["stop"] = ["."]
  return request


def question_transform_input(**request):
  full_prompt = f"""{request["prompt"]}
  Be Concise.
  """
  request["query"] = full_prompt
  return request


def question_transform_output(response):
  # Extract the answer from the responses.
  return str(response)


# This model serving endpoint is created in `02_mlflow_logging_inference`
# question_llm = Databricks(host=workspaceUrl, endpoint_name=maint_chatbot_model_serving_endpoint, transform_input_fn=question_transform_input, transform_output_fn=question_transform_output, model_kwargs={"max_tokens": 300})
question_llm = chain

# COMMAND ----------

def generate_maint_response(message: str,
        # system_prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50):
    
    output = question_llm.run(message)
    return output

# COMMAND ----------

import io
from PIL import Image
import base64

def helper_to_bytes(img):
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    
    return img_bytes.getvalue()
  
one_image = Image.open('./image_inspections/cropped_insulator.png').resize((30,30))
_ = base64.b64encode(helper_to_bytes(one_image)).decode('utf-8')
result = generate_output(_)
result


# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's host it in gradio

# COMMAND ----------

import json
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI

# COMMAND ----------

@dataclass
class ProxySettings:
    proxy_url: str
    port: str
    url_base_path: str


class DatabricksApp:

    def __init__(self, port):
        # self._app = data_app
        self._port = port
        import IPython
        self._dbutils = IPython.get_ipython().user_ns["dbutils"]
        self._display_html = IPython.get_ipython().user_ns["displayHTML"]
        self._context = json.loads(self._dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
        # need to do this after the context is set
        self._cloud = self.get_cloud()
        # create proxy settings after determining the cloud
        self._ps = self.get_proxy_settings()
        self._fastapi_app = self._make_fastapi_app(root_path=self._ps.url_base_path.rstrip("/"))
        self._streamlit_script = None
        # after everything is set print out the url

    def _make_fastapi_app(self, root_path) -> FastAPI:
        fast_api_app = FastAPI(root_path=root_path)

        @fast_api_app.get("/")
        def read_main():
            return {
                "routes": [
                    {"method": "GET", "path": "/", "summary": "Landing"},
                    {"method": "GET", "path": "/status", "summary": "App status"},
                    {"method": "GET", "path": "/dash", "summary": "Sub-mounted Dash application"},
                ]
            }

        @fast_api_app.get("/status")
        def get_status():
            return {"status": "ok"}

        return fast_api_app

    def get_proxy_settings(self) -> ProxySettings:
        if self._cloud.lower() not in ["aws", "azure"]:
            raise Exception("only supported in aws or azure")

        org_id = self._context["tags"]["orgId"]
        org_shard = ""
        # org_shard doesnt need a suffix of "." for dnsname its handled in building the url
        if self._cloud.lower() == "azure":
            org_shard_id = int(org_id) % 20
            org_shard = f".{org_shard_id}"
        cluster_id = self._context["tags"]["clusterId"]
        url_base_path = f"/driver-proxy/o/{org_id}/{cluster_id}/{self._port}"

        from dbruntime.databricks_repl_context import get_context
        host_name = get_context().workspaceUrl
        proxy_url = f"https://{host_name}/driver-proxy/o/{org_id}/{cluster_id}/{self._port}/"

        return ProxySettings(
            proxy_url=proxy_url,
            port=self._port,
            url_base_path=url_base_path
        )

    @property
    def app_url_base_path(self):
        return self._ps.url_base_path

    def mount_gradio_app(self, gradio_app):
        import gradio as gr
        # gradio_app.queue()
        gr.mount_gradio_app(self._fastapi_app, gradio_app, f"/gradio")
        # self._fastapi_app.mount("/gradio", gradio_app)
        self.display_url(self.get_gradio_url())

    def get_cloud(self):
        if self._context["extraContext"]["api_url"].endswith("azuredatabricks.net"):
            return "azure"
        return "aws"

    def get_gradio_url(self):
        # must end with a "/" for it to not redirect
        return f'<a href="{self._ps.proxy_url}gradio/">Click to go to Gradio App!</a>'

    def display_url(self, url):
        self._display_html(url)

    def run(self):
        print(self.app_url_base_path)
        uvicorn.run(self._fastapi_app, host="0.0.0.0", port=self._port)

# COMMAND ----------

import gradio as gr
import random
import time
import base64

DESCRIPTION = f"""
# Chatbot powered by Databricks
This assistant will help you with all of your field maintenance needs
"""


# https://www.gradio.app/docs/gradio/blocks

import numpy as np
import gradio as gr


def maint_question(message: str):
    # system_prompt, max_new_tokens, temperature, top_p, top_k
    output = generate_maint_response(message)
    return output


def part_lookup(img):
    output = generate_output(base64.b64encode(helper_to_bytes(img)).decode('utf-8'))
    return output


with gr.Blocks() as demo:
    gr.Markdown("Your friendly field maintenance assistant!")
    with gr.Tab("Understand how to fix assets"):
        text_input = gr.Textbox(label="Enter your question")
        text_output = gr.Textbox(label="Answer")
        text_button = gr.Button("lookup")
    with gr.Tab("Lookup part details based on an image"):
        with gr.Row():
            image_input = gr.Image(type='pil')
            image_output = gr.Textbox(label="Part Details")
        image_button = gr.Button("lookup")

    

    text_button.click(maint_question, inputs=text_input, outputs=text_output)
    image_button.click(part_lookup, inputs=image_input, outputs=image_output)




# COMMAND ----------

app_port = 8764

# COMMAND ----------

cluster_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().clusterId().getOrElse(None)
workspace_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().workspaceId().getOrElse(None)

print(f"Use this URL to access the chatbot app: ")
print(f"https://dbc-dp-{workspace_id}.cloud.databricks.com/driver-proxy/o/{workspace_id}/{cluster_id}/{app_port}/gradio/")

# COMMAND ----------

dbx_app = DatabricksApp(app_port)

# demo.queue()
dbx_app.mount_gradio_app(demo)

import nest_asyncio
nest_asyncio.apply()
dbx_app.run()

# COMMAND ----------


