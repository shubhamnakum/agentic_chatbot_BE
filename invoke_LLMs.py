import json
import os
import boto3
import logging
from dotenv import load_dotenv
from config import MODEL_Config
load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Bedrock runtime client with credentials
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)


# Invoke LLM classifier:
def invoke_llm(prompt, agent_role):

    model_config = MODEL_Config.get(agent_role)

    model_id = model_config["model_id"]
    model_provider = model_config["provider"]

    if model_provider == "meta":
        result = invoke_Meta(prompt, model_id)
    if model_provider == "deepseek":
        result = invoke_Deepseek(prompt, model_id)
    if model_provider == "mistral":
        result = invoke_Mistral(prompt, model_id)

    return result


# Invoking Deepseek Family models:
def invoke_Deepseek(prompt, model_id):
    # Embed the prompt in DeepSeek-R1's instruction format.
    formatted_prompt = f"<|begin▁of▁sentence|>{prompt}<|Assistant|>\n"
    body = json.dumps({
        "prompt": formatted_prompt,
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 1.0
    })
    model_id = model_id
    # Invoke the model with the request.
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json")
    # Read the response body.
    model_response = json.loads(response["body"].read())

    # Extract choices.
    choices = model_response["choices"]
    print(f"len of choices : {len(choices)}, type: {type(choices)}")
    print(choices)
    final_output = choices[0]["text"].strip().lower()
    logger.info("----------- Deepseek Response Start -----------")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Response: {final_output}")
    logger.info("------------ Deepseek Response End ------------ \n")
    return final_output


# Invoking Meta Family Models:
def invoke_Meta(prompt, model_id):
    formatted_prompt = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    request = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 256,
        "temperature": 0.6,
    })

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=request,
        contentType="application/json",
        accept="application/json")

    model_response = json.loads(response["body"].read())
    response_text = model_response["generation"]
    logger.info("----------- Meta Response Start -----------")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Response: {response_text}")
    logger.info("------------ Meta Response End ------------  \n")
    return response_text


# Invoking Mistral AI Family Models:
def invoke_Mistral(prompt, model_id):
    formatted_prompt = f""" <s>[INST] {prompt} [/INST] """

    request = json.dumps({
        "prompt": formatted_prompt,
        "max_tokens": 512,
        "temperature": 0.4,
        "top_p": 0.7,
        "top_k": 50
    })

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=request,
        contentType="application/json",
        accept="application/json")

    model_response = json.loads(response["body"].read())
    response = model_response.get('outputs')
    response_text = response[0]["text"]
    logger.info("----------- Mistral Response Start -----------")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Response: {response_text}")
    logger.info("------------ Mistral Response End ------------ \n")
    return response_text
