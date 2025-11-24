import json
import argparse
import yaml
import sys


import datetime as dt
import os
from typing import (
    Tuple,
    List,
    Annotated,
    Dict,
    Any
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import List, Dict, Any, Annotated, Tuple
from typing_extensions import TypedDict
import yaml
import os
import logging
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION VARIABLES ---
REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096
AGENTS_DIR = ""
SDDP_AGENT_FILENAME = "agent.yaml" 
SDDP_SCHEMA_FILENAME = "sddp_schema.yaml"

# Global Variables (Loaded via load_sddp_agent_config)
SYSTEM_PROMPT_TEMPLATE: str = ""
USER_PROMPT_SCRIPT_GENERATION: str = ""
SCHEMA_DATA: str = "" # The formatted SDDP Schema content (for injection)

# --- CONFIGURATION LOADING ---

def load_sddp_agent_config(filepath: str) -> bool:
    """Loads the templates and formats the schema content from the new YAML structure."""
    global SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_SCRIPT_GENERATION, SCHEMA_DATA
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
    
def load_sddp_schema(filepath: str) -> bool:
    """
    Load sddp_schema.yaml data and save it on SCHEMA_DATA.
    """
    global SCHEMA_DATA
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            schema_data_dict = yaml.safe_load(file)
            
        SCHEMA_DATA = yaml.dump(schema_data_dict, indent=2, default_flow_style=False, sort_keys=False)
        
        logger.info(f"SDDP Schema loaded and formatted successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error loading or parsing the SDDP schema at {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in load_sddp_schema: {e}")
        return False

# Function to load config once (Simulates agent initialization)
def load_agents_config() -> Dict[str, Any]:
    """Loads the SDDP agent templates (from agent.yaml) and the schema (from sddp_schema.yaml)."""
    
    # 1. Load Agent Prompts
    agent_filepath = os.path.join(AGENTS_DIR, SDDP_AGENT_FILENAME)
    if not os.path.exists(agent_filepath):
        agent_filepath = SDDP_AGENT_FILENAME 
        
    config_loaded = load_sddp_agent_config(agent_filepath)
    if not config_loaded:
        return {'status': 'failed', 'reason': 'Agent Config Failed'}

    # 2. Load SDDP Schema
    schema_filepath = os.path.join(AGENTS_DIR, SDDP_SCHEMA_FILENAME)
    # --- TEMPORARY FIX FOR DEMO: If directory doesn't exist, try local path ---
    if not os.path.exists(schema_filepath):
        schema_filepath = SDDP_SCHEMA_FILENAME # Try the current directory
        
    schema_loaded = load_sddp_schema(schema_filepath)
    if not schema_loaded:
        return {'status': 'failed', 'reason': 'Schema Load Failed'}
    
    # If both loaded successfully
    return {'status': 'loaded'}

# Load configuration once
_AGENTS_CONFIG = load_agents_config()

# --- LANGGRAPH STATE DEFINITION ---

class GraphState(TypedDict):
    """Defines the state for the SDDP script generation workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    input: str        
    schema: str     
    sddp_script: str  
    chat_language: str
    agent_type: str

# --- LANGGRAPH NODES ---

def retrieve_schema(state: GraphState) -> Dict[str, Any]:
    """
    1. Retrieves the pre-loaded, formatted SDDP Schema data.
    """
    logger.info("Node: Retrieving Schema Context")
    
    return {
        "schema": SCHEMA_DATA
    }


def generate_sddp_script(state: GraphState, llm: BaseChatOpenAI) -> Dict[str, Any]:
    """
    2. LLM step: Translates the user request into the structured SDDP Script.
    """
    logger.info("Node: Generating SDDP Script")
    
    # 1. Augment System Prompt (Injection)
    # Replaces the {sddp_schema} placeholder with the retrieved schema content
    system_prompt_content = SYSTEM_PROMPT_TEMPLATE.format(
        sddp_schema=state["schema"]
    )
    system_message = SystemMessage(content=system_prompt_content)
    
    # 2. Create User Prompt
    user_prompt_content = USER_PROMPT_SCRIPT_GENERATION.format(
        user_input=state["input"]
    )
    human_message = HumanMessage(content=user_prompt_content)

    # 3. Invoke LLM
    # 
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
    workflow.add_node("retrieve_schema", retrieve_schema) 
    # 2. Generate SDDP Script (LLM Execution)
    workflow.add_node("generate_sddp_script", generate_sddp_script_partial)
    
    # Defining the flow: Start -> Retrieve Schema -> Generate Script -> End
    workflow.add_edge(START, "retrieve_schema")
    workflow.add_edge("retrieve_schema", "generate_sddp_script")
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
        
        # 2. LangGraph Initialization
        chain, memory = initialize(model)
        logger.info(f"LangGraph Workflow and LLM initialized.")

        # 3. Execution Configuration 
        thread_id = "sddp-script-thread-1" 
        config = {"configurable": {"thread_id": thread_id}}

        # 4. Workflow Execution (2 Steps: Retrieve Schema -> Generate Script)
        logger.info("\n--- EXECUTING AGENT WORKFLOW ---")
        
        result = chain.invoke({
            "input": user_input,
            "messages": [] 
        }, config=config)

        # 5. Processing the Response: Output the raw generated script
        if "sddp_script" in result:
            final_script = result["sddp_script"]
            
            print("\n==============================================")
            print("✅ FINAL SDDP SCRIPT GENERATED:")
            print(final_script)
            print("==============================================\n")
            
        else:
            logger.warning("The workflow was executed, but no SDDP script was generated.")
            
    except Exception as e:
        logger.error(f"\n--- CRITICAL PIPELINE ERROR ---")
        logger.error(f"Error detail: {type(e).__name__}: {str(e)}")
        sys.exit(1)