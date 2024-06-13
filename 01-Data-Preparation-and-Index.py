# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Ingesting and preparing PDF for LLM and Self Managed Vector Search Embeddings
# MAGIC
# MAGIC ## In this example, we will focus on ingesting pdf documents as source for our retrieval process. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-0.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC For this example, we will add Databricks ebook PDFs from [Databricks resources page](https://www.databricks.com/resources) to our knowledge database.
# MAGIC
# MAGIC **Note: This demo is advanced content, we strongly recommend going over the simple version first to learn the basics.**
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use autoloader to load the binary PDFs into our first table. 
# MAGIC - Use the `unstructured` library  to parse the text content of the PDFs.
# MAGIC - Use `llama_index` or `Langchain` to split the texts into chuncks.
# MAGIC - Compute embeddings for the chunks.
# MAGIC - Save our text chunks + embeddings in a Delta Lake table, ready for Vector Search indexing.
# MAGIC
# MAGIC
# MAGIC Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale, including unstructured data like PDFs.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F02-advanced%2F01-PDF-Advanced-Data-Preparation&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-advanced%2F01-PDF-Advanced-Data-Preparation&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-andrew_kraemer` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0108-215105-zlcpmd45/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# TODO make more elegant
dbutils.widgets.text("skip_setup", "false", "Use packages in init script or skip")
skip_setup = dbutils.widgets.get("skip_setup") == "true"

if skip_setup:
  dbutils.notebook.exit("skipped")

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
#TODO centralize packages
%pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.10.1
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingesting Databricks ebook PDFs and extracting their pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC First, let's ingest our PDFs as a Delta Lake table with path urls and content in binary format. 
# MAGIC
# MAGIC We'll use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingeset new files, making it easy to incrementally consume billions of files from the data lake in various data formats. Autoloader easily ingests our unstructured PDF data in binary format.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS raw_document_landing_zone;

# COMMAND ----------

# DBTITLE 1,Download first PDF from Param Link
# List our raw PDF docs
volume_folder =  f"/Volumes/{catalog}/{db}/raw_document_landing_zone"

# get pdf_link from workflow parameter
os.environ['PDF_LINK'] = pdf_link

# COMMAND ----------

# MAGIC %sh cp ./_resources/maint_records.json /tmp/maint_records.json

# COMMAND ----------

dbutils.fs.cp("file:/tmp/maint_records.json", volume_folder)

# COMMAND ----------

spark.sql(f"""create or replace table {catalog}.{db}.asset_locations (location_id string, lat double, long double, location_type string)""")

spark.sql(f"""insert into {catalog}.{db}.asset_locations values
('9',42.0550,-99.2550,"Substation"),
('8F',42.0000,-99.2000,"Transmission_Tower"),
('7C',42.0050,-99.2050,"Transmission_Tower"),
('6D',42.0100,-99.2100,"Transmission_Tower"),
('5B',42.0150,-99.2200,"Transmission_Tower"),
('4A',42.0200,-99.2150,"Transmission_Tower"),
('3C',42.0250,-99.2250,"Transmission_Tower"),
('2B',42.0300,-99.2300,"Transmission_Tower"),
('1D',42.0350,-99.2350,"Transmission_Tower"),
('0C',42.0400,-99.2400,"Transmission_Tower"),
('9A',42.0450,-99.2450,"Transmission_Tower"),

('8',41.9550,-99.1550,"Substation"),
('1E',41.9000,-99.1000,"Transmission_Tower"),
('0B',41.9050,-99.1050,"Transmission_Tower"),
('9D',41.9100,-99.1100,"Transmission_Tower"),
('8A',41.9150,-99.1200,"Transmission_Tower"),
('7C',41.9200,-99.1150,"Transmission_Tower"),
('6B',41.9250,-99.1250,"Transmission_Tower"),
('5D',41.9300,-99.1300,"Transmission_Tower"),
('4C',41.9350,-99.1350,"Transmission_Tower"),
('3A',41.9400,-99.1400,"Transmission_Tower"),
('2E',41.9450,-99.1450,"Transmission_Tower"),

('7',41.8550,-99.0550,"Substation"),
('7F',41.8000,-99.0000,"Transmission_Tower"),
('6C',41.8050,-99.0050,"Transmission_Tower"),
('5D',41.8100,-99.0100,"Transmission_Tower"),
('4B',41.8150,-99.0200,"Transmission_Tower"),
('3A',41.8200,-99.0150,"Transmission_Tower"),
('2C',41.8250,-99.0250,"Transmission_Tower"),
('1B',41.8300,-99.0300,"Transmission_Tower"),
('0D',41.8350,-99.0350,"Transmission_Tower"),
('9C',41.8400,-99.0400,"Transmission_Tower"),
('8A',41.8450,-99.0450,"Transmission_Tower"),

('6',41.7550,-98.9550,"Substation"),
('4E',41.7000,-98.9000,"Transmission_Tower"),
('3B',41.7050,-98.9050,"Transmission_Tower"),
('2D',41.7100,-98.9100,"Transmission_Tower"),
('1A',41.7150,-98.9200,"Transmission_Tower"),
('0C',41.7200,-98.9150,"Transmission_Tower"),
('9B',41.7250,-98.9250,"Transmission_Tower"),
('8D',41.7300,-98.9300,"Transmission_Tower"),
('7C',41.7350,-98.9350,"Transmission_Tower"),
('6A',41.7400,-98.9400,"Transmission_Tower"),
('5E',41.7450,-98.9450,"Transmission_Tower"),

('5',41.6550,-98.8550,"Substation"),
('6F',41.6000,-98.8000,"Transmission_Tower"),
('5C',41.6050,-98.8050,"Transmission_Tower"),
('4D',41.6100,-98.8100,"Transmission_Tower"),
('3B',41.6150,-98.8200,"Transmission_Tower"),
('2A',41.6200,-98.8150,"Transmission_Tower"),
('1C',41.6250,-98.8250,"Transmission_Tower"),
('0B',41.6300,-98.8300,"Transmission_Tower"),
('9D',41.6350,-98.8350,"Transmission_Tower"),
('8C',41.6400,-98.8400,"Transmission_Tower"),
('7A',41.6450,-98.8450,"Transmission_Tower"),

('4',41.5550,-98.7550,"Substation"),
('2E',41.5000,-98.7000,"Transmission_Tower"),
('1B',41.5050,-98.7050,"Transmission_Tower"),
('0C',41.5100,-98.7100,"Transmission_Tower"),
('9D',41.5150,-98.7200,"Transmission_Tower"),
('8A',41.5200,-98.7150,"Transmission_Tower"),
('7B',41.5250,-98.7250,"Transmission_Tower"),
('6C',41.5300,-98.7300,"Transmission_Tower"),
('5D',41.5350,-98.7350,"Transmission_Tower"),
('4A',41.5400,-98.7400,"Transmission_Tower"),
('3E',41.5450,-98.7450,"Transmission_Tower"),

('3',41.4550,-98.6550,"Substation"),
('5F',41.4000,-98.6000,"Transmission_Tower"),
('4C',41.4050,-98.6050,"Transmission_Tower"),
('3D',41.4100,-98.6100,"Transmission_Tower"),
('2B',41.4150,-98.6200,"Transmission_Tower"),
('1A',41.4200,-98.6150,"Transmission_Tower"),
('0C',41.4250,-98.6250,"Transmission_Tower"),
('9B',41.4300,-98.6300,"Transmission_Tower"),
('8D',41.4350,-98.6350,"Transmission_Tower"),
('7C',41.4400,-98.6400,"Transmission_Tower"),
('6A',41.4450,-98.6450,"Transmission_Tower"),

('2',41.3450,-98.5450,"Substation"),
('3B',41.3000,-98.5000,"Transmission_Tower"),
('2D',41.3050,-98.5050,"Transmission_Tower"),
('1E',41.3100,-98.5100,"Transmission_Tower"),
('0A',41.3150,-98.5200,"Transmission_Tower"),
('9D',41.3200,-98.5150,"Transmission_Tower"),
('8E',41.3250,-98.5250,"Transmission_Tower"),
('7B',41.3300,-98.5300,"Transmission_Tower"),
('6C',41.3350,-98.5350,"Transmission_Tower"),
('5A',41.3400,-98.5400,"Transmission_Tower"),
('4D',41.3450,-98.5450,"Transmission_Tower"),

('13',41.2520,-98.4520,"Substation"),
('9B',41.2000,-98.4000,"Transmission_Tower"),
('8C',41.2050,-98.4050,"Transmission_Tower"),
('7G',41.2100,-98.4100,"Transmission_Tower"),
('6D',41.2150,-98.4200,"Transmission_Tower"),
('5E',41.2200,-98.4150,"Transmission_Tower"),
('4A',41.2250,-98.4250,"Transmission_Tower"),
('3C',41.2300,-98.4300,"Transmission_Tower"),
('2B',41.2350,-98.4350,"Transmission_Tower"),
('1D',41.2400,-98.4400,"Transmission_Tower"),
('0E',41.2450,-98.4450,"Transmission_Tower"),


('21',41.1120,-98.3120,"Substation"),
('8E',41.1000,-98.3000,"Transmission_Tower"),
('7F',41.1050,-98.3050,"Transmission_Tower"),
('7D',41.1100,-98.3100,"Transmission_Tower"),
('5B',41.1150,-98.3200,"Transmission_Tower"),
('4E',41.1200,-98.3150,"Transmission_Tower"),
('2A',41.1250,-98.3250,"Transmission_Tower"),
('12B',41.1300,-98.3300,"Transmission_Tower"),
('17E',41.1350,-98.3350,"Transmission_Tower"),
('17A',41.1400,-98.3400,"Transmission_Tower"),
('17B',41.1450,-98.3450,"Transmission_Tower")""")


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-2.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC ## Extracting our PDF content as text chunks
# MAGIC
# MAGIC We need to convert the PDF documents bytes to text, and extract chunks from their content.
# MAGIC
# MAGIC This part can be tricky as PDFs are hard to work with and can be saved as images, for which we'll need an OCR to extract the text.
# MAGIC
# MAGIC Using the `Unstructured` library within a Spark UDF makes it easy to extract text. 
# MAGIC
# MAGIC *Note: Your cluster will need a few extra libraries that you would typically install with a cluster init script.*
# MAGIC
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC ### Splitting our big documentation page in smaller chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC
# MAGIC In this demo, some PDFs are very large, with a lot of text.
# MAGIC
# MAGIC We'll extract the content and then use llama_index `SentenceSplitter`, and ensure that each chunk isn't bigger than 500 tokens. 
# MAGIC
# MAGIC
# MAGIC The chunk size and chunk overlap depend on the use case and the PDF files. 
# MAGIC
# MAGIC Remember that your prompt+answer should stay below your model max window size (4096 for llama2). 
# MAGIC
# MAGIC For more details, review the previous [../01-Data-Preparation](01-Data-Preparation) notebook. 
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to review the chunks created and ensure they make sense and contain relevant information.
# MAGIC </div>

# COMMAND ----------

spark.read.json(f'/Volumes/{catalog}/{db}/raw_document_landing_zone/',multiLine=True).createOrReplaceTempView("raw_records")

# COMMAND ----------

spark.sql(f"""CREATE or replace TABLE {catalog}.{db}.repair_reports (
  id bigint generated always as identity,
  additional_notes STRING,
  component_name STRING,
  issue_identified STRING,
  location_id STRING,
  report_by STRING,
  title STRING,
  challenges STRING,
  content string,
  date DATE)
USING delta
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.enableDeletionVectors' = 'true',
  'delta.feature.deletionVectors' = 'supported',
  'delta.minReaderVersion' = '3',
  'delta.minWriterVersion' = '7')
""")

# COMMAND ----------

spark.sql(f"""insert overwrite table {catalog}.{db}.repair_reports (additional_notes, component_name, issue_identified,location_id, report_by, title, challenges,content, date)  with exploded_reports as(
  select
    explode(reports) as report
  from
    raw_records
),
exploded_elements as(
  select
    report.*
  from
    exploded_reports
)
select
  *
except(
    challenges,
    date,
    recommendations,
    latitude,
    longitude,
    location_type,
    symptoms,
    actions_taken
  ),
  concat_ws(' ', challenges) as challenges,
  'symptoms: '||symptoms||" Actions Taken: "||actions_taken as content,
  to_date(date, 'MMMM d, yyyy') as date
from
  exploded_elements""")

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.repair_reports"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.repair_reports_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_source_column="content",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

# question = "The insulator looks like it has track marks"

# # response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})


# results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
#   query_text=question,
#   columns=["actions_taken", "challenges", "issue_identified"],
#   num_results=1)
# docs = results.get('result', {}).get('data_array', [])
# pprint(docs)

# COMMAND ----------

# question = "This hanger hook hole looks enlarged"

# # response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})


# results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
#   query_text=question,
#   columns=["actions_taken", "challenges", "issue_identified"],
#   num_results=1)
# docs = results.get('result', {}).get('data_array', [])
# pprint(docs)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Self Managed Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02-Advanced-Chatbot-Chain]($./02-Advanced-Chatbot-Chain) notebook to create and deploy a chatbot endpoint.
