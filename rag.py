import argparse
import yaml
import sys
import os
import logging

import datetime as dt
from typing import (
    Tuple,
    List,
    Annotated,
    Dict,
    Any
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION VARIABLES ---
REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096
AGENTS_DIR = ""
SDDP_AGENT_FILENAME = "agent.yaml" 

# Global Variables (Loaded via load_sddp_agent_config)
SYSTEM_PROMPT_TEMPLATE: str = ""
USER_PROMPT_SCRIPT_GENERATION: str = ""


# --- CONFIGURATION LOADING ---

def load_sddp_agent_config(filepath: str) -> bool:
    """Loads the templates and formats the schema content from the new YAML structure."""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_SCRIPT_GENERATION
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        # 1. Load Prompts
        SYSTEM_PROMPT_TEMPLATE = config['prompts']['system_prompt_template'] 
        USER_PROMPT_SCRIPT_GENERATION = config['prompts']['generation_task_prompt']['user_prompt_sddp_generation']
        
        logger.info(f"SDDP Agent templates and schema loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading or parsing the SDDP agent config at {filepath}: {e}")
        return False
    

def load_agent_config() -> Dict[str, Any]:
    """Loads Text-to-Cypher agent configuration."""
    filepath = os.path.join(AGENTS_DIR, SDDP_AGENT_FILENAME)
    if load_sddp_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}

# Load configuration once
_AGENTS_CONFIG = load_agent_config()


# -------------------------------------------------------
# Functions to create State Nodes using LangGraph  
#--------------------------------------------------------

class GraphState(TypedDict):
    """Defines the state for the SDDP script generation workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    input: str        
    examples: str
    properties: str     
    sddp_script: str  
    chat_language: str
    agent_type: str


# -------------------------------------------------------
# Step 1 : Retrive Context 
#--------------------------------------------------------
def get_vectorstore_directory(doc_type: str) -> str:
    return f'vectorstores/{doc_type}'

def load_vectorstore(doc_type: str) -> Chroma:
    """1.1. Loads the vectorstore with examples to use as context """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    persist_directory = get_vectorstore_directory(doc_type)

    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
    raise ValueError(f"Vectorstore not found : {persist_directory}")


def format_contrastive_examples(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes both Correct and Incorrect Codes.
    """
    formatted_blocks = []
    
    for i, doc in enumerate(docs):
        # 1. Get metadata
        metadata = doc.metadata
        
        
        # 3. Create example
        block = f"""
        ### EXample {i + 1}
        Question: {doc.page_content}

        CORRECT SINTAX (DO): ({metadata.get('correct_inst', 'N/A')})
        ```python
        {metadata.get('correct_code', 'N/A')}````

        INCORRECT SINTAX (DON'T DO): ({metadata.get('incorrect_code', 'N/A')})"""

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

def format_properties(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes availables properties
    """
    formatted_blocks = []
    
    available_objects = "The available objects to created with CREATE are the following: ACInterconnection, Area, Battery, Bus, BusShunt, Circuit, CircuitFlowConstraint, CSP, DCBus, DCLine, Demand, DemandSegment, Emission, FlowController, Fuel, FuelConsumption, FuelContract, FuelProducer, FuelReservoir, GasEmission, GasNode, GasPipeline, GenerationConstraint, GenericConstraint, HydroGenerator, HydroPlant, HydroPlantConnection, HydroStation, HydroStationConnection, Interconnection, InterpolationGenericConstraint, LCCConverter, LineReactor, Load, MTDCLink, PaymentSchedule, PowerInjection, RenewableCapacityProfile, RenewableGenerator, RenewablePlant, RenewableStation, RenewableTurbine, RenewableWindSpeedPoint, ReserveGeneration, ReservoirSet, SensitivityGroup, SeriesCapacitor, StaticVarCompensator, SumOfCircuits, SumOfInterconnections, SupplyChainDemand, SupplyChainDemandSegment, SupplyChainFixedConverter, SupplyChainFixedConverterCommodity, SupplyChainNode, SupplyChainProcess, SupplyChainProducer, SupplyChainStorage, SupplyChainTransport, SynchronousCompensator, System, TargetGeneration, ThermalCombinedCycle, ThermalGenerator, ThermalPlant, ThreeWindingsTransformer, Transformer, TransmissionLine, TwoTerminalDCLink, VSCConverter, Waterway, Zone"
    
    formatted_blocks.append(available_objects)

    for i, doc in enumerate(docs):
        # 1. Get metadata
        metadata = doc.metadata
        
        # 2. Extract cypher examples
        objct_name = doc.page_content
        
        # 3. Create example
        block = f"""
        Object: {objct_name}

        Madatory properties to create {objct_name}: {metadata.get("mandatory")}

        Reference properties : {metadata.get("references_objects")}

        Static properties  : {metadata.get("static_properties")}

        Dynamic properties  : {metadata.get("dynamic_properties")}
        """

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

def retrive_properties(state:GraphState)->Dict[str,Any]:
    """
    1.1 Get Factory Properties
    """    
    doc_type= f"properties"
    vectorstore = load_vectorstore(doc_type)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(state["input"])
    properties_str = format_properties(docs)
    print(properties_str)
    return {"properties": properties_str}

def retrive_examples(state:GraphState)->Dict[str,Any]:
    """
    1.2 Get Code examples
    """
    try: 
        doc_type = "examples"
        vectorstore = load_vectorstore(doc_type)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(state["input"])
        examples_str = format_contrastive_examples(docs)
    except: 
        examples_str = ""

    print(examples_str)
    return {"examples": examples_str}


def retrieve_context(state: GraphState) -> Dict[str, Any]:
    """
    1.2. Get general context from documentation
    """
    properties = retrive_properties(state)
    examples = retrive_examples(state)

    return {"examples": examples, "properties":properties}


# -------------------------------------------------------
# Step 2 : Generate SDDP Script
#--------------------------------------------------------

def generate_sddp_script(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    2. LLM step: Translates the user request into the structured SDDP Script.
    """
    logger.info("Node: Generating SDDP Script")
    
    # 1.Replaces the {sddp_schema} placeholder with the retrieved schema content
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        examples=state["examples"],
        properties= state["properties"]
    )
    system_message = SystemMessage(content=system_prompt_content)
    
    # 2. Create User Prompt
    user_prompt_content = USER_PROMPT_SCRIPT_GENERATION.format(
        user_input=state["input"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 3. Invoke LLM 
    response = llm.invoke([system_message, human_message])
    
    sddp_script = response.content.strip()
    
    logger.info(f"SDDP Script generated: {sddp_script.splitlines()[0]}...")
    
    # The final output is the generated script itself
    return {
        "sddp_script": sddp_script,
        "messages": [AIMessage(content=sddp_script)]
    }


# --- LANGGRAPH WORKFLOW ---

def create_langgraph_workflow(llm: BaseChatOpenAI):
    """Creates the LangGraph workflow for the SDDP agent."""
    
    # Partial functions to inject the LLM into the nodes
    def generate_sddp_script_partial(state: GraphState):
        return generate_sddp_script(state, llm)
        
    workflow = StateGraph(GraphState)
    
    # 1. Retrieve Schema (Context Injection)
    workflow.add_node("retrieve_context", retrieve_context) 
    # 2. Generate SDDP Script (LLM Execution)
    workflow.add_node("generate_sddp_script", generate_sddp_script_partial)
    
    # Defining the flow: Start -> Retrieve Schema -> Generate Script -> End
    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_sddp_script")
    workflow.add_edge("generate_sddp_script", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app, memory


def initialize(model: str) -> Tuple[StateGraph, MemorySaver]:
    """Initializes the LLM based on the model name and creates the LangGraph workflow."""
    try:
        # Use a generic mock LLM if you are just testing the workflow flow
        llm = None
        
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(
                model_name="gpt-5-2025-08-07",
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "gpt-4.1":
            llm = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(
                model_name="gpt-4.1-mini",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "o3":
            llm = ChatOpenAI(
                model_name="o3",
                request_timeout=REQUEST_TIMEOUT
            )
        elif model == "claude-4-sonnet":
            llm = ChatAnthropic(
                model='claude-sonnet-4-20250514',
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT
            )
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(
                model='deepseek-reasoner',
                openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                openai_api_base='https://api.deepseek.com',
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )
        else:
            # Default fallback LLM
            llm = ChatOpenAI(
                model_name="gpt-4.1",
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                request_timeout=REQUEST_TIMEOUT
            )

        app, memory = create_langgraph_workflow(llm)
        
        return app, memory

    except Exception as e:
        logger.error(f"Error initializing RAG: {str(e)}")
        raise


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SDDP Script Generation Agent - Translates natural language into SDDP scripts.")
    parser.add_argument("-m", "--model", default="gpt-4.1", 
                        help="LLM model to use (e.g., gpt-4.1, claude-4-sonnet). Default: gpt-4.1")
    parser.add_argument("-q", "--query", required=True, 
                        help="The natural language request to be processed by the agent.")
    
    args = parser.parse_args()

    model = args.model
    user_input = args.query
    
    logger.info(f"--- Starting SDDP Script Generation Agent ---")
    logger.info(f"Selected Model: {model}")
    logger.info(f"User Request: '{user_input}'")

    try:
        # 1. Check Configuration Load Status (RAG Retrieval Step)
        if _AGENTS_CONFIG.get('status') != 'loaded':
            logger.error("Agent configuration and schema failed to load.")
            raise Exception("Configuration load failed.")

        logger.info(f"SDDP Schema loaded from config file.")

        # 2.Download latest RAG
         
        
        # 3. LangGraph Initialization
        chain, memory = initialize(model)
        logger.info(f"LangGraph Workflow and LLM initialized.")

        # 4. Execution Configuration 
        thread_id = "sddp-script-thread-1" 
        config = {"configurable": {"thread_id": thread_id}}

        # 5. Workflow Execution (2 Steps: Retrieve Schema -> Generate Script)
        logger.info("\n--- EXECUTING AGENT WORKFLOW ---")
        
        result = chain.invoke({
            "input": user_input,
            "messages": [] 
        }, config=config)

        # 5. Processing the Response: Output the raw generated script
        if "sddp_script" in result:
            final_script = result["sddp_script"]
            
            print("\n==============================================")
            print("FINAL SDDP SCRIPT GENERATED:")
            print(final_script)
            print("==============================================\n")
            
        else:
            logger.warning("The workflow was executed, but no SDDP script was generated.")
            
    except Exception as e:
        logger.error(f"\n--- CRITICAL PIPELINE ERROR ---")
        logger.error(f"Error detail: {type(e).__name__}: {str(e)}")
        sys.exit(1)