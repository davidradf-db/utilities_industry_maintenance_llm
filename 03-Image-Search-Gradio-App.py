# Databricks notebook source
# MAGIC %pip install --upgrade gradio==3.38.0 fastapi==0.104 uvicorn==0.24
# MAGIC %pip install typing-extensions==4.8.0 --upgrade
# MAGIC %pip install -q -U langchain==0.0.319
# MAGIC %pip install --force-reinstall databricks-genai-inference==0.1.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false $use_old_langchain=true

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
llm = Databricks(host=workspaceUrl, endpoint_name=chatbot_model_serving_endpoint,
                 transform_input_fn=transform_input, transform_output_fn=transform_output)

# COMMAND ----------

def generate_output(img: str):
    
    output = llm.invoke(img)
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

displayHTML(result)

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
This chatbot helps you answers questions regarding {customer_name}. It uses retrieval augmented generation to infuse data relevant to your question into the LLM and generates an accurate response.
"""

def process_example(img):

    # system_prompt, max_new_tokens, temperature, top_p, top_k
    output = generate_output(base64.b64encode(helper_to_bytes(img)).decode('utf-8'))
    return output

demo = gr.Interface(
    fn=process_example,
    inputs=[
        gr.Image(label='test', type='pil'),
            ],
    outputs=[
        gr.Textbox(label="part number")],
)

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


