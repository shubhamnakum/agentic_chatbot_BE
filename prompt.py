from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Few-shot examples for domain classification
examples = [
    {"query": "What are the major revenue sources of Netflix?",
        "agent_type": "retrieval"},
    {"query": "Symptoms of diabetes", "agent_type": "retrieval"},
    {"query": "What is United Airline's cargo revenue?", "agent_type": "retrieval"},
    {"query": "Energy consumption trends of Chevron", "agent_type": "retrieval"},
    {"query": "How does Walmart's supply chain work?", "agent_type": "retrieval"},
    {"query": "What is the net income of Apple in 2024?", "agent_type": "retrieval"},
    {"query": "Best programming languages for AI", "agent_type": "researcher"},
    {"query": "How to apply for scholarships?", "agent_type": "researcher"},
    {"query": "What is weather in London?", "agent_type": "weather"},
    {"query": "What is current temperature in Ahmedabad?", "agent_type": "weather"},
    {"query": "How to learn Python?", "agent_type": "researcher"},
    {"query": "Who is the Prime Minister of India?", "agent_type": "researcher"},
    {"query": "what is the Hemoglobin A1c?", "agent_type": "researcher"},
    {"query": "Common treatments for thyroid issues", "agent_type": "retrieval"},
    {"query": "Does Google have a 10-K filing for 2024?", "agent_type": "retrieval"},
    {"query": "Impact of AI on stock trading", "agent_type": "researcher"}
]

example_prompt = PromptTemplate(
    input_variables=["query", "agent_type"],
    template="Query: {query}\nAgent_Type: {agent_type}\n"
)

classifier_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="""
        You are a strict classifier that selects which type of agent should handle a user's query.

        You have access to documents in the following categories and topics below,  You do NOT have access to documents outside these topics:

        1. **Finance** - 10-K reports for companies including:
        - Amazon, Apple, Microsoft, Google, Meta, Netflix, Nvidia, Tesla, and others.

        2. **Healthcare** - WHO question-answer reports on:
        - Covid, Diabetes, Tuberculosis, Thyroid, Leukemia

        3. **Aviation** - Data related to:
        - American Airlines, Delta Airlines, United Airlines (includes cargo and ESG data)

        4. **Energy** - Data for:
        - Chevron, Exxon, Valero

        5. **Retail** - Data for:
        - Costco, Walmart

        Your task is to classify the query into one of the following agents:

        **Agents**:
        - retrieval: Use this if the query is about any of the companies, industries, or health topics listed above.
        - researcher: Use this if the query is about topics NOT covered by the available documents and may require general web research.
        - weather: Use this if the query is about the weather in any city.

        **Rules**:
        - Be precise and confident.
        - DO NOT explain your reasoning.
        - DO NOT include anything except one of the following lowercase words as your final output: `retrieval` / `researcher` / `weather`
        """,
    suffix="Query: {query}\nAgent_Type:",
    input_variables=["query"],
)


researcher_prompt = PromptTemplate(
    template="""
                You are a research agent. Create a response based on these web search results:
                
                User Query: {query}
                
                Search Results:
                {combined_results}
                
                Instructions:
                1. Synthesize the information directly from the search results
                2. Only include information that appears in the search results
                3. Cite the sources of information (URLs) when possible
                4. If the search results don't contain specific information asked in the query, acknowledge this limitation
                5. Keep your response factual and concise
                6. For financial data or statistics, include the source and date of the information
                """
)

researcher_fallback_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
            You are a research agent. The web search failed for this query: {query}
            
            Provide a general response, but clearly indicate that:
            1. You don't have access to specific up-to-date information about this topic
            2. The user should verify any important details from official sources
            
            IMPORTANT: Do not provide specific statistics, numbers, or recent facts that you aren't certain about.
            """
)

extract_city_prompt = PromptTemplate(
    template="""
        Extract the city name from the following query if it contains a location related to weather.
        If no city is found, respond with 'Unknown'.
        
        just give city name as output.
        
        User Query: {query}
        """
)

writer_prompt = PromptTemplate(
    template="""
        You are a writing agent.
        Using only the information provided in the retrieved documents, write a Detailed and well-structured response to the user query.
        Understand the User Query and the context of the retrieved documents and write a Breif response on that.
        Do not verify or fact-check the retrieved documents.
        Do not provide a one line/ one word answer.
        Only reference the source of the exact document or chunk that contains the retrieved information. Not any other sources.
        Maintain an Positive, informative tone and avoid repetition or filler content.
        Do NOT provide any warnings about the content.

        Respond only in valid JSON format without commentary. Use Format-1 or Format-2 as instructed. Do not include markdown or extra text.

        Relevant Information:
        {retrieved_docs}

        User Query:
        {query}    
        
        Agent Type: {agent_type}            
        
        Respond in valid JSON format using one of these two formats based on the agent_type:
        
        For agent_type "retrieval":
        {{"answer": "your response here", "source": "source information here"}}
        
        For all other agent_type:
        {{"answer": "your response here"}}
        
        Ensure your response is properly formatted JSON with escaped quotes and no trailing commas.
        """
)

critic_prompt = PromptTemplate(
    template="""
        You are a critique and refinement agent.
        Your goal is to enhance the following draft response to make it more detailed, informative, and reader-friendly.
        
        Carefully read the original text, and then:
        - Expand the response where appropriate by elaborating on brief or vague statements
        - Do not add new facts or external information, Do not include commentary or explanation.
        - Maintain the original intent and meaning
        - Ensure a helpful, professional, and positive tone
        - Do not respond with JSON or any structured format — just the improved plain text.
        
        Do NOT include commentary, markdown, or any surrounding text — only return the improved version in plain text.

        Draft Response:
        {draft_response}
        """
)
