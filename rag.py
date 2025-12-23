import argparse
import yaml
import os
import operator
from typing import Tuple, List, Annotated, Dict, Any
import psr.factory

import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.tools import tool
from langchain_chroma import Chroma
import chromadb.config
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

REQUEST_TIMEOUT = 120
MAX_TOKENS = 4096

# Agent template settings
# single-agent config file at repository root
AGENT_CONFIG_PATH = "agent.yaml"


# -----------------------------
# Load agent configuration
# -----------------------------

def load_agent_config(filepath: str) -> bool:
    """Load prompts and templates from the agent YAML configuration.

    Expected structure (agent.yaml):
    prompts:
      system_prompt_template: |
        ...
      prepare_prompt: |
        ...
      generation_task_prompt:
        user_prompt_sddp_generation: |
          ...
    """
    global SYSTEM_PROMPT_TEMPLATE, GENERATION_TASK_PROMPT, PREPARE_PROMPT
    SYSTEM_PROMPT_TEMPLATE = None
    GENERATION_TASK_PROMPT = None
    PREPARE_PROMPT = None
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        prompts = config.get('prompts', {}) if isinstance(config, dict) else {}
        # system prompt
        SYSTEM_PROMPT_TEMPLATE = prompts.get('system_prompt_template') or prompts.get('system_prompt')
        # prepare prompt (for the prepare step)
        PREPARE_PROMPT = prompts.get('prepare_prompt')
        # generation task prompt (nested mapping, for the execution step)
        gen = prompts.get('generation_task_prompt') or {}
        if isinstance(gen, dict):
            GENERATION_TASK_PROMPT = gen.get('user_prompt_sddp_generation')
        elif isinstance(gen, str):
            GENERATION_TASK_PROMPT = gen

        if not SYSTEM_PROMPT_TEMPLATE:
            raise KeyError('system_prompt_template not found in agent config')

        logger.info(f"Agent template loaded successfully from {filepath}")
        return True
    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        logger.error(f"Error loading Agent from {filepath}: {e}")
        return False


def load_agents_config() -> Dict[str, Any]:
    """Load agent configuration file from AGENT_CONFIG_PATH."""
    filepath = AGENT_CONFIG_PATH
    if load_agent_config(filepath):
        return {'status': 'loaded'}
    return {'status': 'failed'}


# Load configuration once
load_agents_config()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add] # Add messages to state (humam or ai result)

class RAGAgent:

    def __init__(self, model, tools, system):
        self.system = system
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}

        workflow = StateGraph(AgentState)
        # prepare: build a natural-language reference script (properties + examples)
        workflow.add_node("prepare", self.prepare_reference)
        workflow.add_node("llm", self.call_llm)
        workflow.add_node("retriver", self.take_action)

        # flow: prepare -> llm -> (if tool calls) retriver -> llm -> ...
        workflow.add_edge('prepare', 'llm')
        workflow.add_conditional_edges(
            'llm',
            self.exists_action,
            {True: 'retriver', False: END}
        )
        workflow.add_edge('retriver', 'llm')
        workflow.set_entry_point('prepare')
        self.workflow = workflow.compile()

    def exists_action(self,state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls)>0 
    
    def call_llm(self, state: AgentState):
        messages = state['messages'] 
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)  # AI response
        return {'messages': [message]}

    def prepare_reference(self, state: AgentState):
        """
        Build a natural-language reference script the agent will use while invoking tools.
        Uses external retriever functions (retrive_properties, retrive_examples)
        to fetch RAG context, then invokes the LLM to generate a reference script
        using PREPARE_PROMPT from agent.yaml.
        
        Returns: Both the reference script AND a follow-up prompt instructing the LLM
        to execute the plan using tools (from GENERATION_TASK_PROMPT).
        """
        try:
            # last user input
            last_user_input = state['messages'][-1].content if state.get('messages') else ''

            # Retrieve properties and examples using external retrievers
            try:
                vectorstore = load_vectorstore("properties")
                retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
                last_message_content = state["messages"][-1].content
                docs = retriever.invoke(last_message_content)
                properties_str = format_properties(docs)
            except Exception as e:
                logger.warning(f"Failed to retrieve properties: {e}")
                properties_str = "(unable to load properties)"

            try:
                examples_dict = retrive_examples(state)
                examples_str = examples_dict.get('examples', '(no examples found)')
            except Exception as e:
                logger.warning(f"Failed to retrieve examples: {e}")
                examples_str = "(unable to load examples)"

            # Use PREPARE_PROMPT from agent.yaml to ask the LLM to generate a reference script outline
            if PREPARE_PROMPT:
                # The prepare prompt asks the LLM to outline the case structure before executing
                prompt = PREPARE_PROMPT
            else:
                # Fallback prompt if PREPARE_PROMPT is not loaded
                prompt = (
                    f"Based on the user's request below and the available SDDP objects/properties, "
                    f"create a structured natural-language outline describing the case you will build.\n\n"
                    f"User request: {last_user_input}\n\n"
                    f"Include:\n"
                    f"1. Objects to Create (object type, key properties)\n"
                    f"2. Mandatory References (dependency order)\n"
                    f"3. Property Assignments (static properties for each object)\n\n"
                    f"Format as a clear, readable checklist for the LLM to follow step-by-step."
                )

            messages = [SystemMessage(content=self.system), HumanMessage(content=prompt)] if self.system else [HumanMessage(content=prompt)]
            reference_script = self.model.invoke(messages)
            
            # Now prepare the execution phase: add a follow-up message with GENERATION_TASK_PROMPT
            # This instructs the LLM to use the tools to implement the reference script
            if GENERATION_TASK_PROMPT:
                execution_prompt = GENERATION_TASK_PROMPT
            else:
                execution_prompt = (
                    "You have prepared a natural-language reference script above. "
                    "Now execute the case creation by invoking the available tools (create_objects, set_static_properties, etc.) "
                    "following the plan you outlined."
                )
            
            execution_message = HumanMessage(content=execution_prompt)
            return {'messages': [reference_script, execution_message]}
        except Exception as e:
            logger.error(f"prepare_reference failed: {e}")
            # fallback to empty message so workflow proceeds
            return {'messages': [HumanMessage(content='(prepare_reference failed)')]} 

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls  # use tools 
        results = []
        for t in tool_calls:
            logger.info(f"Calling Tool: {t}")
            if not t['name'] in self.tools:
                logger.warning(f"Tool {t} doesn't exist")
                result = "Incorrect tool name. Retry and select an available tool"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        logger.info("Tools Execution Complete. Back to the model")
        return {'messages': results}
    


def initialize(model: str) :
    """Initialize the LLM and return the compiled LangGraph workflow and memory."""

    try:
        if model == "gpt-5-2025-08-07":
            llm = ChatOpenAI(model_name="gpt-5-2025-08-07", request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1":
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "gpt-4.1-mini":
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "o3":
            llm = ChatOpenAI(model_name="o3", request_timeout=REQUEST_TIMEOUT)
        elif model == "claude-4-sonnet":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model='claude-sonnet-4-20250514', anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'), temperature=0.7, max_tokens=MAX_TOKENS, timeout=REQUEST_TIMEOUT)
        elif model == "deepseek-reasoner":
            llm = BaseChatOpenAI(model='deepseek-reasoner', openai_api_key=os.getenv('DEEPSEEK_API_KEY'), openai_api_base='https://api.deepseek.com', temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)
        elif model == "local_land":
            llm = ChatOpenAI(model_name="qwen3:14b", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT, base_url= "http://10.246.47.184:10000/v1")
        else:
            llm = ChatOpenAI(model_name="gpt-4.1", temperature=0.7, max_tokens=MAX_TOKENS, request_timeout=REQUEST_TIMEOUT)

        return llm
    
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise

# -----------------------------
# Retrive context
# -----------------------------
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

# -----------------------------
# Tools
# -----------------------------

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

def retrive_examples(state:AgentState)->Dict[str,Any]:
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

def format_properties(docs: List) -> str:
    """
    Formats a list of retrieved documents into a context string
    that includes availables properties
    """
    formatted_blocks = []
    
    available_objects = "The available objects to be used at study.find() or study.create() are the following: ACInterconnection, Area, Battery, Bus, BusShunt, Circuit, CircuitFlowConstraint, CSP, DCBus, DCLine, Demand, DemandSegment, Emission, FlowController, Fuel, FuelConsumption, FuelContract, FuelProducer, FuelReservoir, GasEmission, GasNode, GasPipeline, GenerationConstraint, GenericConstraint, HydroGenerator, HydroPlant, HydroPlantConnection, HydroStation, HydroStationConnection, Interconnection, InterpolationGenericConstraint, LCCConverter, LineReactor, Load, MTDCLink, PaymentSchedule, PowerInjection, RenewableCapacityProfile, RenewableGenerator, RenewablePlant, RenewableStation, RenewableTurbine, RenewableWindSpeedPoint, ReserveGeneration, ReservoirSet, SensitivityGroup, SeriesCapacitor, StaticVarCompensator, SumOfCircuits, SumOfInterconnections, SupplyChainDemand, SupplyChainDemandSegment, SupplyChainFixedConverter, SupplyChainFixedConverterCommodity, SupplyChainNode, SupplyChainProcess, SupplyChainProducer, SupplyChainStorage, SupplyChainTransport, SynchronousCompensator, System, TargetGeneration, ThermalCombinedCycle, ThermalGenerator, ThermalPlant, ThreeWindingsTransformer, Transformer, TransmissionLine, TwoTerminalDCLink, VSCConverter, Waterway, Zone"
    
    formatted_blocks.append(available_objects)

    for i, doc in enumerate(docs):

        # 1. Get object name and metadata (properties)
        metadata = doc.metadata
        objct_name = doc.page_content
        
        # 3. Create example
        block = f"""
        Object Name: {objct_name}

        Madatory properties to create {objct_name}: {metadata.get("mandatory")}

        Reference properties wich must be used to link objects: {metadata.get("references_objects")}

        Static properties which can be acessed with .get(PropertyName) function and created by .set(PropertyName,value) function : {metadata.get("static_properties")}

        Dynamic properties which can be acessed with .get_df(PropertyName) or .get_at(PropertyName, date) functions and created by .set_df(df) 
        or .set_at(PropertyName, date, value) function : {metadata.get("dynamic_properties")}
        """

        formatted_blocks.append(block.strip())
        
    return "\n\n" + "\n\n".join(formatted_blocks)

@tool
def retrive_properties(state:AgentState)->str:
    """
    Retrieve detailed information about available object types and their properties from the SDDP study.

    Tool behavior:
    - Input: `state` (AgentState) where `state['messages'][-1]` contains the user's query or context.
    - Output: a single string containing a concise, machine-readable description of:
      * mandatory properties for each relevant object type
      * reference properties (names and expected target types)
      * static and dynamic properties and how to set them

    Usage guidance for the LLM (important):
    - Call this tool BEFORE attempting to create objects.
    - Use the property names and reference keys returned here EXACTLY when invoking
      `create_objects` or `set_static_properties`.
    - The returned text is formatted for programmatic consumption; prefer extracting
      exact property names and examples rather than paraphrasing.

    Returns: Formatted documentation of available objects and their properties.
    """
    vectorstore = load_vectorstore("properties")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    last_message_content = state["messages"][-1].content
    docs = retriever.invoke(last_message_content)
    properties_str = format_properties(docs)
    return properties_str


def check_basic_properties(objtype, code, name, id):
    # Code check
    if not isinstance(code, int) or code > 99:
        return "Code must be an integer lower than 99"
    if len(STUDY.find_by_code(objtype, code)) > 0:
        return f"Code {code} already registered for {objtype}. Choose a new code"
    # Name check
    if not isinstance(name, str) or len(name) > 12:
        return "Name must be a string with 12 characters maximum"
    # ID check
    if not isinstance(id, str) or len(id) > 2:
        return "Id must be a string with 2 characters maximum"
    if len(STUDY.find_by_id(objtype, id)) > 0:
        return f"ID {id} is already been used for {objtype}. Choose a different one"

    return True

def get_correspondent_objct_type(reftype:str):
    mapping = {
        "RefDemand": "Demand",
        "RefBus": "Bus",
        "RefSegment": "DemandSegment",
        "RefGenerators": ["ThermalGenerator","HydroGenerator","RenewableGenerator"],
        "RefCircuit": "TransmissionLine",
        "RefArea": "Area",
        "RefZone": "Zone",
        "RefSystem": "System",
        "RefPlants": ["ThermalPlant", "HydroPlant", "RenewablePlant"],
        "RefFuels":"Fuel",
        "RefGasNode":"GasNode",
        "RefNodes":"GasNode",
        "RefStation": ["RenewableStation","HydroStation"],
        "RefStations": "HydroStation",
        "RefInterconnections": "Interconnection",
        "RefControlledBus": "Bus",
        "RefCircuits": ["TransmissionLine","Transformer","ACInterconnection","SeriesCapacitor"],
        "RefBuses": ["DCBus","Bus"],
        "RefLink": "MTDCLink",
        "RefReservoirs": "FuelReservoir",
        "RefFuelReservoirs": "FuelReservoir",
        "RefFuelProducers": "FuelProducer",
        "RefSegment": "SupplyChainDemandSegment",
        "RefProcess": "SupplyChainProcess",
    }
    # Return mapped object type; fall back to removing leading 'Ref' if unknown
    if reftype in mapping:
        return mapping[reftype]
    if isinstance(reftype, str) and reftype.startswith("Ref"):
        return reftype[3:]
    return reftype

def check_mandatory_refs(refs:Dict[str,list]):
    # If there are no refs or empty dict, nothing to check
    if not refs:
        return True
    for reftype, codes in refs.items():
        objtypes = get_correspondent_objct_type(reftype)
        # normalize to list of candidate types
        candidates = objtypes if isinstance(objtypes, (list, tuple)) else [objtypes]
        # allow single int or list of codes
        if codes is None:
            return f"Reference {reftype} provided without codes"
        codes = codes if isinstance(codes, (list, tuple)) else [codes]
        for code in codes:
            # check if any candidate type has the object with given code
            found = False
            for candidate in candidates:
                objs = STUDY.find_by_code(candidate, code)
                if len(objs) > 0:
                    found = True
                    break
            if not found:
                return f"Referenced object not found: type_candidates={candidates}, code={code}"
    return True

@tool 
def create_objects(object_type:str, name:str, id:str, code:int, refs:Dict[str,list]): 
    """
    Create a new study object in the SDDP `STUDY` graph.

    Tool behavior:
    - Inputs:
      * `object_type` (str): the exact object type name (use values from `retrive_properties`).
      * `name` (str): user-visible name.
      * `id` (str): short identifier (2 chars max).
      * `code` (int): numeric code (integer < 100).
      * `refs` (Dict[str, list|int]): mapping of reference property names (the keys
         as returned by `retrive_properties`, e.g. `RefPlants`, `RefFuel`) to a single
         code or list of codes referencing existing objects.

    - Output: A short string describing success or a clear error message.

    Important for the LLM:
    - Always call `retrive_properties` first to learn required property names and
      reference keys. Provide `refs` using those keys exactly.
    - If the tool returns an error string, the LLM should modify the input and retry.
    """
    # Validate basic properties first
    basic_check = check_basic_properties(object_type, code, name, id)
    if basic_check is not True:
        return f"Basic property validation failed: {basic_check}"

    # Validate referenced objects
    refs_check = check_mandatory_refs(refs)
    if refs_check is not True:
        return f"Reference validation failed: {refs_check}"

    # Create the object
    try:
        obj = STUDY.create(object_type)
        obj.name = name
        obj.code = code
        obj.id = id
    except Exception as e:
        return f"Failed to create object {object_type}: {e}"

    # Set references if any
    if refs:
        for reftype, codes in refs.items():
            try:
                obj.set(reftype, codes)
            except Exception as e:
                return f"Failed to set reference {reftype} on {object_type} (code={code}): {e}"
            
    STUDY.add(obj)

    return f"Created {object_type} with code={code} id={id}"
    
def check_object_exist(objtyppe,code):
    obj = STUDY.find_by_code(objtyppe,code)
    if len(obj)==0 :
        return "Object of type and code doens't exists"
    
    return True


@tool 
def set_static_properties(objtype, code, properties:Dict[str,Any]):
    """
    Set one or more static properties on an existing object.

    Tool behavior:
    - Inputs:
      * `objtype` (str): exact object type name
      * `code` (int): object code to find the existing object
      * `properties` (Dict[str, Any]): mapping property_name -> value
    - Validates object existence and that each property is static (not a dataframe).
    - Returns a success message or a clear error string.

    Usage guidance for the LLM:
    - Use property names exactly as returned by `retrive_properties`.
    - If a property is dynamic (time-series), call `set_dynamic_properties` instead.
    """
    check_object_exist(objtype,code)
    obj = STUDY.find_by_code(objtype,code)[0]
    for prop, value in properties.items():
        description = obj.description(prop)
        is_dataframe = description.is_dynamic() and len(description.dimensions()) > 0
        if is_dataframe:
            return "Can't create dataframe with this function. Please use set_dynamic_properties"
        obj.set(prop, value)
    return f"Set static properties on {objtype} code={code}"

@tool
def save_study():
    """
    Persist the current `STUDY` to disk.

    Tool behavior:
    - No inputs.
    - Side-effect: saves study to `./study_llm_test` (currently hard-coded).
    - Output: success string or exception message.

    Usage guidance for the LLM:
    - Call this tool when all objects/properties have been created and you want to
      persist the study.
    """
    study_path = "./study_llm_test"
    psr.factory.save_study(study_path)
    return "Study saved with success"



if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="RAG Agent - Translate questions to Cypher and answer them with RAG + auto-fix tools")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="LLM model to be used. Default: gpt-4.1")
    #parser.add_argument("-s", "--study_path", required=True, help="Path to the study files required to load the graph schema into Neo4j.")
    parser.add_argument("-q", "--query", required=True, help="Natural language question to be processed by the agent.")
    args = parser.parse_args()
    model = args.model
    #study_path = args.study_path
    user_input = args.query


    logger.info("--- Starting RAG Agent ---")
    logger.info(f"Selected model: {model}")
    #logger.info(f"Study path: {study_path}")
    logger.info(f"User question: {user_input}")
    try:
        global STUDY
        STUDY = psr.factory.create_study()
        logger.info("Study loaded successfully.")
        
        llm = initialize(model)
        logger.info("LLM initialized.")
        
        # Register available tools for the agent. Include all tool functions the
        # LLM might call during the workflow.
        tools = [
            retrive_properties,
            create_objects,
            set_static_properties,
            save_study,
        ]

        # Initialize agent with system prompt and user query
        initial_message = HumanMessage(content=user_input)
        messages = [initial_message]
        
        # Create agent with system prompt (as string, not list)
        agent = RAGAgent(llm, tools, SYSTEM_PROMPT_TEMPLATE)
        logger.info("Agent initialized with tools and system prompt.")
        
        # Invoke workflow
        result = agent.workflow.invoke({'messages': messages})
        logger.info("Workflow execution completed.")
        
        # Extract and display final response
        if "messages" in result and result["messages"]:
            final_response = result["messages"][-1].content
            print("\n==============================================")
            print("AGENT FINAL RESPONSE:")
            print(final_response)
            print("==============================================\n")
        else:
            logger.warning("Workflow executed but no answer generated.")
    except Exception as e:
        logger.error("--- CRITICAL ERROR ---")
        logger.error(f"Error Detail: {type(e).__name__}: {str(e)}")
        sys.exit(1)
