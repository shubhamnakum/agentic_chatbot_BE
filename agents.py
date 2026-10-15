from typing import Dict, TypedDict, List
from config import CHROMA_PERSIST_DIR, Weather_API_KEY, Embedding_Model
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import logging
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from duckduckgo_search import DDGS
from invoke_LLMs import invoke_llm
from dotenv import load_dotenv
from prompt import *
import ast
import json
import time
import random
from serpapi import GoogleSearch
import os

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
embedding_function = HuggingFaceEmbeddings(model_name=Embedding_Model)


# Define the state schema
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    current_input: str
    retrieved_docs: list[str]
    draft_response: str
    final_response: str
    retrieval_successful: bool
    category: str
    source: str
    city: str
    cleaned_docs: List[Dict[str, str]]
    domain: List[str]


def create_retrieval_agent() -> AgentState:
    vectordb = Chroma(
        collection_name="AllSectors",
        embedding_function=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectordb


def extract_clean_documents(documents):
    """Extract and clean documents from the retrieved results."""
    cleaned = []

    for item in documents:
        doc = item[0] if isinstance(item, tuple) else item
        source = doc.metadata.get('source')
        raw = doc.page_content

        # parsing stringified list chunks
        parts = [part.strip() for part in raw.split('|') if part.strip()]
        content_lines = []

        for part in parts:
            try:
                parsed = ast.literal_eval(part)
                if isinstance(parsed, list):
                    # Filter out empty strings and None
                    cleaned_lines = [line for line in parsed if line]
                    content_lines.extend(cleaned_lines)
            except Exception:
                content_lines.append(part.strip())

        final_text = '\n'.join(content_lines).strip()

        cleaned.append({
            "source": source,
            "content": final_text
        })
    return cleaned


def retriever(state: AgentState) -> AgentState:
    """Retrieve documents and update the state."""
    try:
        domain = state["domain"]
        query = state["current_input"]
        vectordb = create_retrieval_agent()

        examples = [
            {"query": "Cargo specifications for American Airlines", "domain": "aviation"},
            {"query": "Door clearance dimensions for Boeing 777", "domain": "aviation"},
            {"query": "Chevron's 2024 energy production data", "domain": "energy"},
            {"query": "Valero's fuel efficiency report", "domain": "energy"},
            {"query": "Diabetes treatment clinical studies", "domain": "healthcare"},
            {"query": "Thyroid cancer research papers", "domain": "healthcare"},
            {"query": "JPMC 2024 financial performance", "domain": "finance"},
            {"query": "Bank of America quarterly earnings", "domain": "finance"},
            {"query": "Costco retail sales analytics", "domain": "retail"},
            {"query": "Walmart supply chain optimization", "domain": "retail"},
            {"query": "Apple's 10x revenue projections", "domain": "finance"},
        ]

        example_prompt = PromptTemplate(
            input_variables=[query, domain],
            template="Query: {query}\nDomain: {domain}\n"
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Classify the domain for the following query. Choose from: aviation, energy, healthcare, finance, or retail. Respond with the domain name only. DO NOT include anything else like extra characters or symbols.",
            suffix="Query: {input}\nDomain:",
            input_variables=["input"],
        )

        def classify_domain(query: str) -> list[str]:
            """Use LLM to classify domain if not explicit."""
            prompt = few_shot_prompt.format(input=query)
            domain_prediction = invoke_llm(prompt, agent_role="classifier")
            domain = domain_prediction.strip().splitlines()[
                0].strip()
            return [domain]

        def domain_based_retrieval_agent(state: AgentState, domain: list[str]) -> str:
            if not domain:
                print("Domain not explicitly mentioned. Classifying...")
                domain = classify_domain(state["current_input"])
            print(f"Classified domain: {domain}")

            # BM25 retriever
            raw = vectordb.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(raw["documents"], raw["metadatas"])
                if m.get("domain") in (domain or [])
            ]

            BM25_retriever = BM25Retriever.from_documents(docs, k=5)
            bm25_docs = BM25_retriever.get_relevant_documents(
                state["current_input"])
            # Vector retriever
            vector_docs_with_scores = vectordb.similarity_search_with_score(
                query, k=5, filter={"domain": {"$in": domain}}
            )
            vector_docs = [doc for doc, _ in vector_docs_with_scores]

            # RRF (Reciprocal Rank Fusion)
            def reciprocal_rank_fusion(results_lists, k=60):
                score_dict = {}
                for results in results_lists:
                    for rank, doc in enumerate(results):
                        doc_id = (doc.page_content, frozenset(
                            doc.metadata.items()))
                        score = 1 / (k + rank + 1)  # rank starts at 0
                        if doc_id in score_dict:
                            score_dict[doc_id]["score"] += score
                        else:
                            score_dict[doc_id] = {"score": score, "doc": doc}
                ranked = sorted(score_dict.values(),
                                key=lambda x: x["score"], reverse=True)
                return [(item["doc"], item["score"]) for item in ranked[:5]]

            reciprocal_docs = reciprocal_rank_fusion([bm25_docs, vector_docs])

            if not reciprocal_docs:
                return "No relevant documents found."

            for i, (doc, score) in enumerate(reciprocal_docs):
                source = doc.metadata.get("source", "Unknown Source")
                print(f"\n--- Chunk {i+1} ---")
                print(f"Source: {source}")
                print(f"score: {score:.4f}")
                # Print first 500 characters
                print(f"Content: {doc.page_content[:500]}")
                print("-" * 80)

            if not reciprocal_docs:
                return "No relevant documents found."
            sources = [doc.metadata.get(
                "source", "Unknown Source") for doc, _ in reciprocal_docs]
            state["source"] = ", ".join(set(sources))

            cleaned_docs = extract_clean_documents(reciprocal_docs)
            retrieved_documents = [
                f"{doc['source']}\n\n{doc['content']}" for doc in cleaned_docs]

            return {
                "retrieved_documents": retrieved_documents,
                "cleaned_docs": cleaned_docs,
            }

        retrieval_result = domain_based_retrieval_agent(state, domain)
        state["retrieved_docs"] = retrieval_result["retrieved_documents"]
        state["cleaned_docs"] = retrieval_result["cleaned_docs"]

    except Exception as e:
        logger.error(f"Error in Retrieval agent: {str(e)}")
        state["retrieved_docs"] = "Error retrieving documents."
        state["source"] = ""

    return state


def classifier(state: AgentState) -> AgentState:
    """Classify user query as 'retrieval', 'researcher', or 'weather'."""
   # state = retriever(state)

    # retrieved_docs = state["retrieved_docs"]
    print("##################")
    prompt = classifier_prompt.format(
        query=state["current_input"],
        # retrieved_docs=retrieved_docs,
        # source=state["source"]
    )
    print("Inside classifier agent")
    result = invoke_llm(prompt, agent_role="classifier")
    state["category"] = result.strip().split()[0]
    print(f"result agent: {state["category"]}")
    return state

def search_serpapi(query):
    api_key = os.getenv("SERPAPI_KEY")
    print("api_key:", api_key, query)
    search = GoogleSearch({
        "q": query,
        "api_key": api_key
    })
    print("search query:", search)
    results = search.get_dict().get("organic_results", [])
    print("results:", results)
    formatted = []

    for r in results[:5]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("link", "")
        formatted.append(f"Title: {title}\nExcerpt: {snippet}\nURL: {url}")

    print("formatted results:", formatted)
    return formatted, results


def researcher(state: AgentState) -> AgentState:
    """Fetch external data through DuckDuckGo web search with rate limit handling, fallback to SerpAPI."""
    query = state["current_input"]
    
    # Retry configuration for DuckDuckGo
    max_retries = 3
    base_delay = 2  # Base delay in seconds
    
    # First, try DuckDuckGo with retries
    ddgs_success = False
    search_results = []
    
    try:
        for attempt in range(max_retries):
            try:
                # Add exponential backoff with jitter
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Rate limited. Waiting {delay:.2f} seconds before retry {attempt + 1}")
                    time.sleep(delay)
                
                with DDGS() as ddgs:
                    # Try different backends and configure DDGS to be more conservative
                    text_results = list(ddgs.text(
                        query, 
                        max_results=3,  # Reduced from 5 to be more conservative
                    ))
                    
                    for result in text_results:
                        title = result.get("title", "")
                        body = result.get("body", "")
                        href = result.get("href", "")
                        search_results.append(
                            f"Title: {title}\nExcerpt: {body}\nURL: {href}\n"
                        )
                    
                    # If we got results, break out of retry loop
                    if search_results:
                        ddgs_success = True
                        break
                        
            except Exception as e:
                error_msg = str(e).lower()
                if 'ratelimit' in error_msg or '202' in str(e):
                    logger.warning(f"DuckDuckGo rate limit hit on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        logger.warning("DuckDuckGo exhausted all retries due to rate limiting")
                    continue
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"DuckDuckGo error on attempt {attempt + 1}: {e}")
                    break

        # If DuckDuckGo was successful, process results
        if ddgs_success and search_results:
            # Combine search results
            combined_results = "\n\n".join(search_results)
            
            # Use LLM to synthesize response
            prompt = researcher_prompt.format(
                query=query,
                combined_results=combined_results
            )
            
            result = invoke_llm(prompt, agent_role=state["category"])
            state["retrieved_docs"] = result
            
            # Extract sources
            sources = [
                result.get("href", "").split("//")[-1].split("/")[0]
                for result in text_results if "href" in result
            ]
            unique_sources = list(set(sources))
            state["source"] = f"Web Search Results ({', '.join(unique_sources[:3])})"
            
            return state
            
    except Exception as e:
        logger.error(f"Unexpected error in DuckDuckGo search: {str(e)}")

    # If DuckDuckGo failed, try SerpAPI
    logger.info("DuckDuckGo failed, attempting SerpAPI...")
    try:
        search_results, raw_results = search_serpapi(query)
        print("search results:", search_results)
        print("raw results:", raw_results)
        
        if search_results:
            combined_results = "\n\n".join(search_results)
            print("Combined Results:", combined_results)
            
            # Use the LLM to synthesize a response based on search results
            prompt = researcher_prompt.format(
                query=query,
                combined_results=combined_results
            )
            print("Prompt for LLM:", prompt)
            result = invoke_llm(prompt, agent_role=state["category"])
            print("LLM Result:", result)
            state["retrieved_docs"] = result

            sources = [r.get("link", "").split("//")[-1].split("/")[0]
                      for r in raw_results if "link" in r]
            print("Sources:", sources)
            unique_sources = list(set(sources))
            print("Unique Sources:", unique_sources)
            state["source"] = f"Web Search Results ({', '.join(unique_sources[:3])})"
            
            return state
        else:
            state["retrieved_docs"] = "No relevant search results found. I don't have specific information about your query."
            state["source"] = "Web Search (No Results)"
            return state

    except Exception as e:
        logger.error(f"Error in SerpAPI search: {str(e)}")

    # If both DuckDuckGo and SerpAPI failed, use fallback
    logger.warning("Both DuckDuckGo and SerpAPI failed, using fallback...")
    prompt = researcher_fallback_prompt.format(query=query)
    result = invoke_llm(prompt, agent_role=state["category"])
    state["retrieved_docs"] = result
    state["source"] = "General Knowledge (Web Search Failed)"

    return state

def extract_city(state: AgentState) -> AgentState:
    """Extract city name from the user query using LLM."""
    prompt = extract_city_prompt.format(query=state["current_input"])
    result = invoke_llm(prompt, agent_role=state["category"])
    state["city"] = result.strip()
    return state


def weather_agent(state: AgentState) -> AgentState:
    """Retrieve weather data based on user-provided city."""
    def get_temperature(api_key, location):
        """Fetch temperature for the given location using WeatherAPI."""
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temperature = data['current']['temp_c']
            return f"The current temperature in {location} is {temperature}°C."
        else:
            return "Unable to fetch the data. Please check the API key and location."

    state = extract_city(state)
    city = state.get("city", "Unknown")
    api_key = Weather_API_KEY
    weather_data = get_temperature(api_key, city)
    state["retrieved_docs"] = weather_data
    state["source"] = f"Weather API -- api.weatherapi.com, (City: {city})"
    print(f"Weather: {state["retrieved_docs"]}")
    return state


def writer(state: AgentState) -> AgentState:
    """Compose a response based on retrieved information."""

    prompt = writer_prompt.format(
        retrieved_docs=state["retrieved_docs"],
        query=state["current_input"],
        agent_type=state["category"]
    )

    raw_output = invoke_llm(prompt, agent_role="writer")

    result = json.loads(raw_output)

    state["draft_response"] = result.get("answer", "")
    if state["category"] == "retrieval":
        state["source"] = result.get("source", "")
    return state


def critic(state: AgentState) -> AgentState:
    """Improve the response before final output."""
    prompt = critic_prompt.format(draft_response=state["draft_response"])
    result = invoke_llm(prompt, agent_role="critic")
    state["final_response"] = result + \
        f"\n\n(Source: {state["source"]})"
    return state
