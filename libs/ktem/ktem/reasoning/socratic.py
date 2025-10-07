import logging
from typing import TypedDict, Optional, Any
import uuid

from ktem.utils.generator import Generator
from ktem.reasoning.base import BaseReasoning
from ktem.reasoning.react import ReactAgentPipeline, DEFAULT_QA_PROMPT
from ktem.llms.manager import llms
from kotaemon.base import Document, BaseComponent
from kotaemon.llms import ChatLLM
from kotaemon.agents import (
    SocraticQuestionerAgent, 
    SocraticEvaluatorAgent, 
    SocraticReflectionAgent
)


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer, interrupt, Command



logger = logging.getLogger(__name__)



class SocraticState(TypedDict):
    conv_id: str
    context: str
    user_query: str  
    socratic_query: str 
    user_response: str
    history: list[dict[str, str]]
    turn_number: int 
    latest_eval: dict[str, Any]
    reflection: str


class SocraticPipeline(BaseReasoning):

    class Config:
        allow_extra = True
 

    def __init__(
            self, 
            react_pipeline: ReactAgentPipeline = ReactAgentPipeline.withx(), 
            questioner_agent: SocraticQuestionerAgent = SocraticQuestionerAgent.withx(), 
            evaluator_agent: SocraticEvaluatorAgent = SocraticEvaluatorAgent.withx(), 
            reflection_agent: SocraticReflectionAgent = SocraticReflectionAgent.withx(), 
            checkpointer: Checkpointer = MemorySaver(), 
            max_turns: int =5):
        super().__init__()
        self.react_pipeline = react_pipeline
        self.questioner_agent = questioner_agent
        self.evaluator_agent = evaluator_agent
        self.reflection_agent = reflection_agent
        self.max_turns = max_turns
        self.checkpointer = checkpointer
        self.graph = self._build_graph(self.checkpointer)



    def _build_graph(self, checkpointer: Checkpointer):

        def decide_next_node(state: SocraticState):
            
            print(f"DEBUG Router - {state}")

            if state["turn_number"] > self.max_turns:
                return "reflect"
            
            next_action = state.get("latest_eval", "continue")
            print(f"NEXT ACTION: {next_action}")
            return next_action



        def generate_context(state: SocraticState) -> SocraticState:
            context_parts = []
            
            for response in self.react_pipeline.stream(
                message=state["user_query"],
                conv_id="",
                history=[]
            ):
                if isinstance(response, Document) and response.channel == "chat":
                    if response.content:
                        context_parts.append(response.content)
            
        
            state["context"] = "".join(context_parts)

            return state
        
        
        def generate_socratic_query(state: SocraticState) -> SocraticState:
            query = self.questioner_agent({
                "context": state["context"], 
                "history": state["history"]
            })

            state ["socratic_query"] = query.text
            state["history"].append({self.questioner_agent.agent_type: query.text})
            state["turn_number"] += 1

            return state 

        
        def get_user_response(state: SocraticState):
            user_response = interrupt(value="")
            state["user_response"] = user_response
            state["history"].append({"user": user_response})

            print(f"GET_USER_RESPONSE: {state}")


            return state
                
        

        def evaluate_user_response(state: SocraticState) -> SocraticState:
            print("EVALUATING")
            state["latest_eval"] = self.evaluator_agent({
                "context": state["context"], 
                "history": state["history"]
            }).text

            return state
        

        def generate_reflection(state: SocraticState) -> SocraticState:
            state["reflection"] = self.reflection_agent({
                "context": state["context"], 
                "history": state["history"]
            }).text

            return state


        graph = StateGraph(SocraticState)

       
        graph.add_node("contextualize", generate_context)
        graph.add_node("question", generate_socratic_query)
        graph.add_node("user_response", get_user_response)
        graph.add_node("evaluate", evaluate_user_response)
        graph.add_node("reflect", generate_reflection)

        graph.add_edge(START, "contextualize")
        graph.add_edge("contextualize", "question")
        graph.add_edge("question", "user_response")
        graph.add_edge("user_response", "evaluate")

        graph.add_conditional_edges(
            "evaluate", 
            decide_next_node, 
            { 
                "continue": "question", 
                "reflect": "reflect"
            }
        )

        graph.add_edge("reflect", END)
        

        compiled_graph = graph.compile(checkpointer=checkpointer)

        return compiled_graph
    


    def stream(self, message, conv_id: str, history: list, pipeline_state: dict = None, **kwargs):
        if self.graph is None:
            raise RuntimeError("Pipeline not properly initialized. Use get_pipeline() classmethod.")

        config = {"configurable": {"thread_id": conv_id, "checkpoint_ns": "socratic"}}
        
        # Check if we're resuming from an interrupt
        is_resuming = pipeline_state and pipeline_state.get("waiting_for_input", False)
        
        if is_resuming:
            # Resume from interrupt with user's response
            print(f"DEBUG: Resuming with message: {message}")
            for chunk in self.graph.stream(Command(resume=message), config, stream_mode="updates"):
                yield from self._process_chunk(chunk)
        else:
            # Start new conversation or continue existing one
            try:
                # Try to get existing state
                current_state = self.graph.get_state(config)
                if current_state and current_state.values:
                    print("DEBUG: Continuing existing conversation")
                    # Don't reinitialize, just resume
                    for chunk in self.graph.stream(None, config, stream_mode="updates"):
                        yield from self._process_chunk(chunk)
                else:
                    print("DEBUG: Starting new conversation")
                    initial_state = SocraticState(
                        conv_id=conv_id,
                        context="", 
                        user_query=message, 
                        socratic_query="",
                        user_response="", 
                        history=[], 
                        turn_number=0, 
                        latest_eval={},
                        reflection=""
                    )
                    
                    for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
                        yield from self._process_chunk(chunk)
            except Exception as e:
                print(f"DEBUG: Error checking state, starting fresh: {e}")
                initial_state = SocraticState(
                    conv_id=conv_id,
                    context="", 
                    user_query=message, 
                    socratic_query="",
                    user_response="", 
                    history=[], 
                    turn_number=0, 
                    latest_eval={},
                    reflection=""
                )
                
                for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
                    yield from self._process_chunk(chunk)


    def _process_chunk(self, chunk):
        """Helper method to process graph chunks"""
        for node_id, value in chunk.items():
            if node_id == "question":
                yield Document(
                    channel="chat",
                    content=value["socratic_query"],
                )
            elif node_id == "reflect":
                yield Document(
                    channel="chat",
                    content=value["reflection"],
                )
            elif node_id == "__interrupt__":
                yield Document(
                    channel="interrupt",
                    content={"waiting_for_input": True}
                )

    # def _save_checkpoint(self, config, state, metadata=None):
    #     checkpoint_id = (
    #         config.get("configurable", {}).get("thread_id")
    #         or state.get("conv_id")
    #         or str(uuid.uuid4())
    #     )

    #     checkpoint = {
    #         "id": checkpoint_id,        
    #         "channel_values": state,   
    #         "channel_versions": {}, 
    #         "next": None,               
    #     }

    #     self.checkpointer.put(
    #         config,
    #         checkpoint,
    #         metadata or {},
    #         new_versions={},
    #     )
    

    # def _save_checkpoint(self, config, state, metadata=None):
    #     """Ensure checkpoint is saved in LangGraph's expected structure."""
    #     from langgraph.checkpoint.base import CheckpointTuple

    #     checkpoint_data = CheckpointTuple(
    #         # the actual pipeline state
    #         channel_values=state,
    #         # optional: version tracking, can be empty
    #         channel_versions={},
    #         # optional: metadata for LangGraph to use internally
    #         metadata=metadata or {},
    #         # optional: the next checkpoint link (not required)
    #         next=None,
    #     )

    #     self.checkpointer.put(config, checkpoint_data)

    # def stream(self, message, conv_id: str, history: list, **kwargs):
    #     if self.graph is None:
    #         raise RuntimeError("Pipeline not properly initialized. Use get_pipeline() classmethod.")

    #     config = {"configurable": {"thread_id": conv_id, "checkpoint_ns": "socratic"}}

    #     initial_state = SocraticState(
    #                         conv_id=conv_id,
    #                         context="", 
    #                         user_query=message, 
    #                         socratic_query="",
    #                         user_response="", 
    #                         history=[], 
    #                         turn_number=0, 
    #                         latest_eval="",
    #                         reflection=""
    #                     ) 
        
    #     for chunk in self.graph.stream(initial_state, config):
    #         for node_id, value in chunk.items():
    #             if node_id == "question":
    #                 yield Document(
    #                     channel="chat",
    #                     content=value["socratic_query"],
    #                 )
    #             elif node_id == "reflect":
    #                 yield Document(
    #                     channel="chat",
    #                     content=value["reflection"],
    #                 )
    #             elif node_id == "__interrupt__":
    #                 # yield Document(
    #                 #     channel="interrupt",
    #                 #     content={"waiting_for_input": True}
    #                 # ) 
    #                 self.graph.invoke(Command(resume=message), config) 

                    

    # def stream(self, message, conv_id: str, history: list, **kwargs):
    #     if self.graph is None:
    #         raise RuntimeError("Pipeline not properly initialized. Use get_pipeline() classmethod.")

    #     # Check if resuming from interrupt
    #     pipeline_state = kwargs.get("pipeline_state", {})
    #     is_interrupted = pipeline_state.get("waiting_for_input", False)
        
    #     config = {"configurable": {"thread_id": conv_id, "checkpoint_ns": "socratic"}}
        
    #     if is_interrupted:
    #         # Resume with user's response
    #         # saved_state = self.checkpointer.get(config)
    #         # if not saved_state:
    #         #     raise RuntimeError("No saved SocraticState found for resume.")

    #         # # Update state with user's reply
    #         # saved_state["user_response"] = message
    #         # saved_state["history"].append({"user": message})
    #         # saved_state["turn_number"] += 1

    #         output_stream = self.graph.stream(
    #             Command(resume=message),
    #             config=config,
    #             stream_mode="updates"
    #         )

    #     else:

    #         initial_state = SocraticState(
    #             conv_id=conv_id,
    #             context="", 
    #             user_query=message, 
    #             socratic_query="",
    #             user_response="", 
    #             history=[], 
    #             turn_number=0, 
    #             latest_eval="",
    #             reflection=""
    #         )

    #         # self._save_checkpoint(config, dict(initial_state))

    #         output_stream = Generator(self.graph.stream(
    #             initial_state,
    #             config=config, 
    #             stream_mode="updates"
    #         ))


    #     for event in output_stream:
    #         if "question" in event:
    #             yield Document(
    #                 channel="chat",
    #                 content=event["question"]["socratic_query"],
    #             )
    #         elif "reflect" in event:
    #             yield Document(
    #                 channel="chat",
    #                 content=event["reflect"]["reflection"],
    #             )
    #         elif "__interrupt__" in event:
    #             yield Document(
    #                 channel="interrupt",
    #                 content={"waiting_for_input": True}
    #             )
        
        # Ensure interrupt flag cleared when done
        # saved_state = self.checkpointer.get(config)
        # if saved_state and not saved_state.get("waiting_for_input", False):
        #     saved_state["waiting_for_input"] = False
        #     # self.checkpointer.put(config, saved_state, metadata={}, new_versions={})
        #     self._save_checkpoint(config, saved_state)
        
    
    
    @classmethod
    def get_pipeline(
        cls, settings: dict, states: dict, retrievers: list | None = None
    ) -> BaseReasoning:
        _id = cls.get_info()["id"]
        prefix = f"reasoning.options.{_id}"

        llm_name = settings[f"{prefix}.llm"]
        llm = llms.get(llm_name, llms.get_default())
        max_turns = settings[f"{prefix}.dialogue_max_turns"]

        react_pipeline = ReactAgentPipeline.get_pipeline(settings=settings, states=states, retrievers=retrievers)
        
        pipeline = SocraticPipeline(
            react_pipeline=react_pipeline, 
            questioner_agent=SocraticQuestionerAgent(llm=llm), 
            evaluator_agent=SocraticEvaluatorAgent(llm=llm), 
            reflection_agent=SocraticReflectionAgent(llm=llm), 
            checkpointer=MemorySaver(),
            max_turns=max_turns
        )

        return pipeline

    @classmethod
    def get_user_settings(cls) -> dict:
        settings = ReactAgentPipeline.get_user_settings()
        settings.update(
            {
                "dialogue_max_turns": {
                    "name": "Maximum Socratic dialogue turns",
                    "value": 5,
                    "component": "number",
                    "info": "Maximum number of Socratic dialogue exchanges before reflection.",
                }
            }
        )

        return settings
    

    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "Socratic",
            "name": "Socratic",
            "description": (
                "Combines a ReAct agent (for factual grounding) with an adaptive "
                "Socratic dialogue loop that asks probing questions and evaluates responses."
            ),
        }
        




# class SocraticPipeline(BaseReasoning):
#     """
#     A hybrid reasoning pipeline that:
#       1. Uses ReactAgentPipeline to form a ground-truth context/answer.
#       2. Runs a Socratic dialogue loop (Questioner ↔ Evaluator ↔ Reflection)
#          based on that context.
#     """

#     def __init__(self, retrievers=None, llm=None, react_settings=None):
#         super().__init__()
#         self.llm = llm or llms.get_default()
#         self.retrievers = retrievers or []
#         self.react_settings = react_settings or {}
#         self.checkpointer = MemorySaver()

#         # Create embedded ReAct pipeline
#         self.react_pipeline = ReactAgentPipeline.get_pipeline(
#             settings=react_settings or {},
#             states={},
#             retrievers=retrievers,
#         )

#         # Create Socratic LangGraph
#         self.socratic_graph = self._create_graph()

#     # -------------------------------------------------------------------------
#     # Graph Setup
#     # -------------------------------------------------------------------------

#     def _create_graph(self):
#         """Create LangGraph state machine for Socratic loop."""
#         graph = StateGraph(dict)

#         def evaluator_to_next(state: dict[str, Any]):
#             feedback = state.get("output", {})
#             if isinstance(feedback, dict) and feedback.get("need_followup", True):
#                 return "continue"
#             return "reflect"

#         graph.add_node("questioner", SocraticQuestionerAgent())
#         graph.add_node("evaluator", SocraticEvaluatorAgent())
#         graph.add_node("reflect", SocraticReflectionAgent())

#         graph.add_edge("questioner", "evaluator")
#         graph.add_conditional_edges(
#             "evaluator", evaluator_to_next,
#             {"continue": "questioner", "reflect": "reflect"}
#         )

#         graph.set_entry_point("questioner")
#         return graph.compile(checkpointer=self.checkpointer)

#     # -------------------------------------------------------------------------
#     # Core Methods
#     # -------------------------------------------------------------------------

#     async def ainvoke(self, message: str, conv_id: str, history: list, **kwargs) -> Document:
#         """
#         Async Socratic reasoning:
#         - Step 1: Get ground-truth answer from ReactAgentPipeline
#         - Step 2: Run Socratic dialogue loop
#         """
#         # --- Step 1: generate ground-truth answer using ReactAgent ---
#         react_result = await self.react_pipeline.ainvoke(message, conv_id, history)
#         ground_truth = react_result.content if isinstance(react_result, Document) else str(react_result)

#         # --- Step 2: run Socratic graph ---
#         state = {
#             "user_message": message,
#             "ground_truth": ground_truth,
#             "history": history,
#         }

#         result = await self.socratic_graph.ainvoke(state)
#         output = result.get("output")

#         # Return as Kotaemon-compatible Document
#         return Document(channel="chat", content=output or "")

#     def stream(self, message: str, conv_id: str, history: list, **kwargs):
#         """
#         Streaming version — combines ReAct pipeline and Socratic dialogue.
#         Kotaemon-compatible generator that yields intermediate steps.
#         """
#         # --- Step 1: Stream ReAct reasoning ---
#         for doc in self.react_pipeline.stream(message, conv_id, history):
#             yield doc
#         ground_truth = self.react_pipeline.agent.last_output_text or ""

#         # --- Step 2: Socratic adaptive questioning ---
#         input_state = {
#             "user_message": message,
#             "ground_truth": ground_truth,
#             "history": history,
#         }

#         output_stream = Generator(self.socratic_graph.stream(input_state))
#         for item in output_stream:
#             output = item.get("output")
#             thought = item.get("thought", "")

#             if isinstance(output, dict) and "evaluation" in output:
#                 yield Document(channel="info", content=f"💡 {output['evaluation']}")
#                 if hint := output.get("hint"):
#                     yield Document(channel="info", content=f"🧭 Hint: {hint}")
#             else:
#                 yield Document(channel="chat", content=output)

#             if thought:
#                 yield Document(channel="debug", content=f"({thought})")

#         return output_stream.value

#     # -------------------------------------------------------------------------
#     # Kotaemon Integration
#     # -------------------------------------------------------------------------

#     @classmethod
#     def get_pipeline(cls, settings: dict, states: dict, retrievers: list | None = None):
#         """Factory method expected by Kotaemon."""
#         _id = cls.get_info()["id"]
#         prefix = f"reasoning.options.{_id}"

#         # Get Socratic LLM
#         llm_name = settings.get(f"{prefix}.llm", None)
#         llm = llms.get(llm_name) if llm_name else llms.get_default()

#         # Extract ReactAgent settings subset (namespaced)
#         react_prefix = f"{prefix}.react"
#         react_settings = {
#             f"{react_prefix}.llm": settings.get(f"{react_prefix}.llm", ""),
#             f"{react_prefix}.max_iterations": settings.get(f"{react_prefix}.max_iterations", 5),
#             f"{react_prefix}.tools": settings.get(f"{react_prefix}.tools", ["SearchDoc", "LLM"]),
#             f"{react_prefix}.qa_prompt": settings.get(f"{react_prefix}.qa_prompt", ""),
#         }

#         return SocraticReasoningPipeline(
#             retrievers=retrievers,
#             llm=llm,
#             react_settings=react_settings,
#         )

#     @classmethod
#     def get_user_settings(cls) -> dict:
#         """
#         Combine Socratic pipeline options + ReactAgent options
#         so users can configure both in the Settings UI.
#         """
#         llm_choices = [("(default)", "")] + [(k, k) for k in llms.options().keys()]
#         tool_choices = ["Wikipedia", "Google", "LLM", "SearchDoc"]

#         return {
#             "llm": {
#                 "name": "Language Model (Socratic)",
#                 "value": "",
#                 "component": "dropdown",
#                 "choices": llm_choices,
#                 "info": (
#                     "The language model used for Socratic questioning and evaluation. "
#                     "If None, defaults to the app’s main LLM."
#                 ),
#             },
#             # React pipeline settings
#             "react.llm": {
#                 "name": "Ground Truth Generator LLM",
#                 "value": "",
#                 "component": "dropdown",
#                 "choices": llm_choices,
#             },
#             "react.tools": {
#                 "name": "Tools for Knowledge Retrieval",
#                 "value": ["SearchDoc", "LLM"],
#                 "component": "checkboxgroup",
#                 "choices": tool_choices,
#             },
#             "react.max_iterations": {
#                 "name": "ReAct Max Iterations",
#                 "value": 5,
#                 "component": "number",
#             },
#             "react.qa_prompt": {
#                 "name": "ReAct QA Prompt",
#                 "value": "",
#                 "component": "textarea",
#             },
#         }

#     @classmethod
#     def get_info(cls) -> dict:
#         return {
#             "id": "Socratic",
#             "name": "Socratic Reasoning",
#             "description": (
#                 "Combines a ReAct agent (for factual grounding) with an adaptive "
#                 "Socratic dialogue loop that asks probing questions and evaluates responses."
#             ),
#         }





# class SocraticState(TypedDict):
#     """State tracked throughout the Socratic dialogue"""
#     # User input
#     student_reply: str
#     initial_question: str
    
#     # Context
#     ground_truth_context: str
    
#     # Agent outputs
#     socratic_questions: str
#     evaluator_decision: str
#     hint_content: str
#     reflection_content: str
    
#     # Orchestrator state
#     turn_number: int
#     explored_concepts: list
#     dialogue_history: list
#     understanding_level: str
    
#     # Flow control
#     next_action: str
#     final_answer: str
    
#     # Metadata
#     orchestrator_metadata: dict
#     socratic_metadata: dict
#     evaluator_metadata: dict
    
#     # Kotaemon integration
#     conversation_id: str
#     is_first_turn: bool


# class SocraticPipeline(BaseReasoning):
#     """
#     Enhanced Socratic tutoring pipeline with LangGraph orchestration.
#     Fully integrated with Kotaemon's reasoning framework.
    
#     Flow:
#     1. get_ground_truth: ReAct pipeline retrieves context from RAG
#     2. initialize_dialogue: Orchestrator initializes state (first turn only)
#     3. ask_socratic_questions: Socratic agent generates questions
#     4. wait_for_student: Returns questions to user, waits for response
#     5. evaluate_response: Evaluator assesses student understanding
#     6. orchestrate: Orchestrator decides next action
#     7. Route to: continue (back to 3), hint, or reflect
#     """
    
#     class Config:
#         allow_extra = True

#     retrievers: list[BaseComponent] = []
#     react_pipeline: Optional[ReactAgentPipeline] = None

#     def __init__(
#         self, 
#         llm: Optional[ChatLLM] = None,
#         max_turns: int = 8,
#         stuck_threshold: int = 3,
#         retrievers: Optional[list] = None,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.llm = llm or llms.get_default()
#         self.max_turns = max_turns
#         self.stuck_threshold = stuck_threshold
#         self.retrievers = retrievers or []

#         # Initialize ReAct pipeline for RAG (will be configured by get_pipeline)
#         if self.react_pipeline is None:
#             self.react_pipeline = ReactAgentPipeline(llm=self.llm, retrievers=self.retrievers)
        
#         # Initialize Socratic agents
#         self.orchestrator = SocraticOrchestratorAgent(
#             llm=self.llm,
#             max_turns=max_turns,
#             stuck_threshold=stuck_threshold
#         )
#         self.socratic_agent = SocraticQuestionerAgent(llm=self.llm)
#         self.evaluator_agent = SocraticEvaluatorAgent(llm=self.llm)
#         self.reflection_agent = SocraticReflectionAgent(llm=self.llm)

#         # Session storage: maps conversation_id to orchestrator state
#         self._sessions = {}

#         # Build graph
#         self.graph = self._build_graph()

#     def _get_or_create_session(self, conv_id: str):
#         """Get or create orchestrator for this conversation"""
#         if conv_id not in self._sessions:
#             self._sessions[conv_id] = {
#                 "orchestrator": SocraticOrchestratorAgent(
#                     llm=self.llm,
#                     max_turns=self.max_turns,
#                     stuck_threshold=self.stuck_threshold
#                 ),
#                 "initialized": False,
#             }
#         return self._sessions[conv_id]

#     def _clear_session(self, conv_id: str):
#         """Clear session after dialogue ends"""
#         if conv_id in self._sessions:
#             del self._sessions[conv_id]

#     # --- LangGraph Node Functions ---
    
#     def get_ground_truth(self, state: SocraticState) -> dict:
#         """Use ReAct pipeline to get ground truth context from RAG"""
#         try:
#             # Only get ground truth on first turn
#             if state.get("is_first_turn"):
#                 logger.info("Getting ground truth from ReAct pipeline...")
                
#                 # Use ReAct pipeline which has RAG tools configured
#                 # Pass any stored kwargs (file_ids, settings, etc.)
#                 kwargs = getattr(self, '_current_kwargs', {})
                
#                 # Call ReAct pipeline using stream and collect result
#                 result_content = ""
#                 for chunk in self.react_pipeline.stream(
#                     message=state["initial_question"],
#                     conv_id=state.get("conversation_id", "default"),
#                     history=[],
#                     **kwargs  # Pass through file_ids, settings, etc.
#                 ):
#                     # Collect chat responses (ignore info messages)
#                     if hasattr(chunk, 'channel') and chunk.channel == "chat":
#                         if hasattr(chunk, 'content'):
#                             result_content += chunk.content
#                         elif hasattr(chunk, 'text'):
#                             result_content += chunk.text
                
#                 context = result_content.strip() if result_content else "No context retrieved"
#                 logger.info(f"Retrieved ground truth context: {context[:100]}...")
                
#                 # Clean up kwargs
#                 if hasattr(self, '_current_kwargs'):
#                     delattr(self, '_current_kwargs')
                
#                 return {"ground_truth_context": context}
#             else:
#                 # Reuse existing context
#                 logger.info("Reusing existing ground truth context")
#                 return {}
#         except Exception as e:
#             logger.error(f"Error in get_ground_truth: {e}", exc_info=True)
#             return {"ground_truth_context": "Error retrieving context. Please ensure documents are uploaded."}

#     def initialize_dialogue(self, state: SocraticState) -> dict:
#         """Initialize orchestrator state on first turn"""
#         try:
#             if not state.get("is_first_turn"):
#                 return {}  # Skip if not first turn
                
#             logger.info("Initializing dialogue...")
#             conv_id = state.get("conversation_id", "default")
#             session = self._get_or_create_session(conv_id)
#             orchestrator = session["orchestrator"]
            
#             # Initialize orchestrator
#             result = orchestrator.run({
#                 "action": "init",
#                 "ground_truth_context": state.get("ground_truth_context", "")
#             })
            
#             session["initialized"] = True
#             logger.info("Dialogue initialized")
            
#             return {
#                 "turn_number": 1,
#                 "explored_concepts": [],
#                 "dialogue_history": [],
#                 "understanding_level": "partial",
#                 "orchestrator_metadata": result.metadata or {},
#             }
#         except Exception as e:
#             logger.error(f"Error in initialize_dialogue: {e}", exc_info=True)
#             return {}

#     def ask_socratic_questions(self, state: SocraticState) -> dict:
#         """Socratic agent generates questions"""
#         try:
#             logger.info("Generating Socratic questions...")
#             conv_id = state.get("conversation_id", "default")
#             session = self._get_or_create_session(conv_id)
#             orchestrator = session["orchestrator"]
            
#             # Determine questioning strategy
#             understanding = state.get("understanding_level", "partial")
#             turn_num = state.get("turn_number", 1)
            
#             strategy = "standard"
#             if turn_num >= self.stuck_threshold and understanding == "confused":
#                 strategy = "hint"
#             elif understanding == "strong":
#                 strategy = "synthesis"
            
#             logger.info(f"Using strategy: {strategy}, understanding: {understanding}")
            
#             # Prepare context for Socratic agent
#             context_dict = orchestrator.prepare_agent_context(
#                 state.get("student_reply", ""),
#                 "socratic"
#             )
#             context_dict.update({
#                 "understanding_level": understanding,
#                 "questioning_strategy": strategy,
#                 "student_analysis": state.get("evaluator_metadata", {}).get("student_analysis", ""),
#             })
            
#             result = self.socratic_agent.run(context_dict)
            
#             if result.status == "failed":
#                 logger.error(f"Socratic agent failed: {result.error}")
#                 return {"socratic_questions": "Can you explain your thinking?"}
            
#             logger.info(f"Generated questions: {result.text[:100]}...")
            
#             return {
#                 "socratic_questions": result.text,
#                 "socratic_metadata": result.metadata or {},
#                 "final_answer": result.text,  # Return to user
#             }
#         except Exception as e:
#             logger.error(f"Error in ask_socratic_questions: {e}", exc_info=True)
#             return {"socratic_questions": "What are your thoughts on this?"}

#     def evaluate_response(self, state: SocraticState) -> dict:
#         """Evaluator assesses student's response"""
#         try:
#             logger.info("Evaluating student response...")
#             context_dict = {
#                 "context": state.get("ground_truth_context", ""),
#                 "student_reply": state.get("student_reply", ""),
#                 "history": state.get("dialogue_history", []),
#                 "turn_number": state.get("turn_number", 1),
#                 "max_turns": self.max_turns,
#             }
            
#             result = self.evaluator_agent.run(context_dict)
            
#             if result.status == "failed":
#                 logger.error(f"Evaluator failed: {result.error}")
#                 return {"evaluator_decision": "continue_socratic"}
            
#             logger.info(f"Evaluation decision: {result.text}")
            
#             return {
#                 "evaluator_decision": result.text,
#                 "evaluator_metadata": result.metadata or {},
#                 "understanding_level": result.metadata.get("understanding_level", "partial") if result.metadata else "partial",
#             }
#         except Exception as e:
#             logger.error(f"Error in evaluate_response: {e}", exc_info=True)
#             return {"evaluator_decision": "continue_socratic"}

#     def orchestrate(self, state: SocraticState) -> dict:
#         """Orchestrator decides next action"""
#         try:
#             logger.info("Orchestrating next action...")
#             conv_id = state.get("conversation_id", "default")
#             session = self._get_or_create_session(conv_id)
#             orchestrator = session["orchestrator"]
            
#             # Create mock evaluator output
#             class EvaluatorOutput:
#                 def __init__(self, text, metadata):
#                     self.text = text
#                     self.metadata = metadata
            
#             evaluator_output = EvaluatorOutput(
#                 state.get("evaluator_decision", "continue_socratic"),
#                 state.get("evaluator_metadata", {})
#             )
            
#             # Build orchestrator instruction
#             instruction = {
#                 "action": "next_action",
#                 "evaluator_output": evaluator_output,
#                 "socratic_metadata": state.get("socratic_metadata", {}),
#                 "student_reply": state.get("student_reply", ""),
#                 "agent_response": state.get("socratic_questions", ""),
#             }
            
#             result = orchestrator.run(instruction)
            
#             if result.status == "failed":
#                 logger.error(f"Orchestrator failed: {result.error}")
#                 return {"next_action": "reflect"}
            
#             logger.info(f"Orchestrator decision: {result.text}")
            
#             return {
#                 "next_action": result.text,
#                 "turn_number": result.metadata.get("turn_number", state.get("turn_number", 1)) if result.metadata else state.get("turn_number", 1),
#                 "orchestrator_metadata": result.metadata or {},
#                 "explored_concepts": result.metadata.get("state_summary", {}).get("explored_concepts", []) if result.metadata else [],
#             }
#         except Exception as e:
#             logger.error(f"Error in orchestrate: {e}", exc_info=True)
#             return {"next_action": "reflect"}

#     def give_hint(self, state: SocraticState) -> dict:
#         """Provide a hint to the student"""
#         try:
#             # Create a helpful hint based on context and student's confusion
#             hint_prompt = f"""Based on the context below, provide a brief, helpful hint (2-3 sentences) that guides the student without giving away the answer.

# Ground Truth Context:
# {state.get('ground_truth_context', '')}

# Student's Latest Response:
# {state.get('student_reply', '')}

# Recent Dialogue:
# {self._format_dialogue_for_hint(state.get('dialogue_history', []))}

# Provide a hint that:
# 1. Points them toward a key concept they're missing
# 2. Doesn't solve the problem directly
# 3. Encourages them to think about a specific aspect

# Hint:"""
            
#             from kotaemon.llms import PromptTemplate
#             hint_template = PromptTemplate(template=hint_prompt)
#             result = self.llm(hint_template.populate())
            
#             hint = result.text if hasattr(result, "text") else str(result)
#             hint = hint.strip()
            
#             logger.info("Providing hint to student")
            
#             # Clear session after hint
#             conv_id = state.get("conversation_id", "default")
#             self._clear_session(conv_id)
            
#             return {
#                 "hint_content": hint,
#                 "final_answer": f"💡 **Hint**\n\n{hint}\n\nThink about this and feel free to ask another question to continue learning!",
#             }
#         except Exception as e:
#             logger.error(f"Error in give_hint: {e}")
#             return {
#                 "hint_content": "Consider reviewing the key concepts.",
#                 "final_answer": "💡 **Hint**: Consider reviewing the key concepts and try thinking about the problem from a different angle."
#             }
    
#     def _format_dialogue_for_hint(self, history: list) -> str:
#         """Format recent dialogue for hint generation"""
#         if not history:
#             return "No prior dialogue."
        
#         # Get last 4 exchanges
#         recent = history[-8:] if len(history) > 8 else history
#         formatted = []
#         for entry in recent:
#             if isinstance(entry, dict):
#                 role = entry.get("role", "unknown")
#                 content = entry.get("content", "")
#                 formatted.append(f"{role.capitalize()}: {content}")
        
#         return "\n".join(formatted)

#     def reflect(self, state: SocraticState) -> dict:
#         """Provide reflection and summary"""
#         try:
#             conv_id = state.get("conversation_id", "default")
#             session = self._get_or_create_session(conv_id)
#             orchestrator = session.get("orchestrator")
            
#             # Prepare rich context for reflection
#             context_dict = {
#                 "history": state.get("dialogue_history", []),
#                 "context": state.get("ground_truth_context", ""),
#                 "explored_concepts": state.get("explored_concepts", []),
#                 "final_understanding": state.get("understanding_level", "partial"),
#                 "metadata": {
#                     "turn_count": state.get("turn_number", 0),
#                     "orchestrator_state": orchestrator.get_state_summary() if orchestrator else {}
#                 }
#             }
            
#             result = self.reflection_agent.run(context_dict)
            
#             if result.status == "failed":
#                 logger.error(f"Reflection failed: {result.error}")
#                 reflection = "Great work exploring this topic!"
#             else:
#                 reflection = result.text
            
#             logger.info("Providing reflection to student")
#             logger.info(f"Reflection metadata: {result.metadata}")
            
#             # Clear session after reflection
#             self._clear_session(conv_id)
            
#             return {
#                 "reflection_content": reflection,
#                 "final_answer": f"{reflection}",  # No emoji prefix, it's in the reflection
#             }
#         except Exception as e:
#             logger.error(f"Error in reflect: {e}")
#             return {
#                 "reflection_content": "Great work!",
#                 "final_answer": "🎓 Great work exploring this topic!"
#             }

#     # --- Routing Functions ---
    
#     def route_after_orchestration(self, state: SocraticState) -> str:
#         """Route based on orchestrator decision"""
#         next_action = state.get("next_action", "reflect")
        
#         if next_action == "continue_socratic":
#             return "socratic"
#         elif next_action == "hint":
#             return "hint"
#         else:  # reflect
#             return "reflect"

#     def check_if_first_turn(self, state: SocraticState) -> str:
#         """Route based on whether this is first turn"""
#         if state.get("is_first_turn"):
#             return "get_context"
#         else:
#             return "evaluate"

#     # --- Graph Construction ---
    
#     def _build_graph(self):
#         """Build the LangGraph workflow"""
#         graph = StateGraph(SocraticState)

#         # Add nodes
#         graph.add_node("get_ground_truth", self.get_ground_truth)
#         graph.add_node("initialize", self.initialize_dialogue)
#         graph.add_node("socratic", self.ask_socratic_questions)
#         graph.add_node("evaluate", self.evaluate_response)
#         graph.add_node("orchestrate", self.orchestrate)
#         graph.add_node("hint", self.give_hint)
#         graph.add_node("reflect", self.reflect)

#         # Set entry point
#         graph.set_entry_point("get_ground_truth")

#         # Define edges
#         graph.add_edge("get_ground_truth", "initialize")
#         graph.add_edge("initialize", "socratic")
        
#         # After socratic questions, check if first turn
#         graph.add_conditional_edges(
#             "socratic",
#             self.check_if_first_turn,
#             {
#                 "get_context": END,  # First turn: return questions and wait
#                 "evaluate": "evaluate",  # Subsequent turns: evaluate
#             }
#         )
        
#         graph.add_edge("evaluate", "orchestrate")
        
#         # Route based on orchestrator decision
#         graph.add_conditional_edges(
#             "orchestrate",
#             self.route_after_orchestration,
#             {
#                 "socratic": END,  # Return to user, wait for response
#                 "hint": "hint",
#                 "reflect": "reflect",
#             }
#         )
        
#         graph.add_edge("hint", END)
#         graph.add_edge("reflect", END)

#         return graph.compile()

#     # --- Kotaemon Integration ---
    
#     def stream(self, message: str, conv_id: str, history: list[dict], **kwargs):
#         """
#         Streaming entry point for Kotaemon (synchronous generator).
        
#         Args:
#             message: User's current message
#             conv_id: Conversation ID for session tracking
#             history: Conversation history from Kotaemon
#             **kwargs: Additional context (file_ids, settings, etc.)
            
#         Yields:
#             Document objects with incremental responses
#         """
#         logger.info(f"SocraticPipeline.stream called with message: {message[:50]}...")
#         logger.info(f"kwargs: {kwargs.keys() if kwargs else 'None'}")
        
#         try:
#             # Determine if this is the first turn
#             session = self._get_or_create_session(conv_id)
#             is_first = not session.get("initialized", False)
            
#             logger.info(f"Is first turn: {is_first}, conv_id: {conv_id}")
            
#             # Show different status based on turn
#             if is_first:
#                 yield Document(content="🔍 Retrieving relevant information from your notes...", channel="info")
#             else:
#                 yield Document(content="🤔 Evaluating your response...", channel="info")
            
#             # Build initial state
#             initial_state = {
#                 "student_reply": message,
#                 "initial_question": message if is_first else "",
#                 "conversation_id": conv_id,
#                 "is_first_turn": is_first,
#                 "ground_truth_context": kwargs.get("context", ""),
#                 "turn_number": session.get("orchestrator").turn_number if not is_first else 1,
#                 "dialogue_history": session.get("orchestrator").dialogue_history if not is_first else [],
#                 "explored_concepts": session.get("orchestrator").explored_concepts if not is_first else [],
#             }
            
#             logger.info(f"Initial state created, is_first_turn: {initial_state['is_first_turn']}")
            
#             # Store kwargs for ReAct pipeline
#             if is_first:
#                 self._current_kwargs = kwargs
#                 logger.info(f"Stored kwargs for ReAct pipeline")
            
#             # Show progress
#             if is_first:
#                 yield Document(content="💭 Preparing Socratic questions...", channel="info")
            
#             # Execute the graph SYNCHRONOUSLY
#             logger.info("Invoking LangGraph...")
#             result = self.graph.invoke(initial_state)
#             logger.info(f"LangGraph completed. Result keys: {result.keys()}")
            
#             # Get final answer
#             final_answer = result.get("final_answer", "Let's continue exploring this topic.")
#             logger.info(f"Final answer: {final_answer[:100]}...")
            
#             # Stream the final response
#             yield Document(content=final_answer, channel="chat")
            
#         except Exception as e:
#             logger.error(f"Error in Socratic pipeline stream: {e}", exc_info=True)
#             yield Document(
#                 content=f"I encountered an error: {str(e)}\n\nPlease try again or rephrase your question.",
#                 channel="chat"
#             )
    
#     def run(self, message: str, conv_id: str, history: list[dict], **kwargs) -> Document:
#         """Synchronous entry point for Kotaemon"""
#         import asyncio
        
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         try:
#             result = loop.run_until_complete(
#                 self.ainvoke(message, conv_id, history, **kwargs)
#             )
#             return result
#         finally:
#             loop.close()

#     async def ainvoke(self, message: str, conv_id: str, history: list[dict], **kwargs) -> Document:
#         """
#         Async entry point for Kotaemon runtime.
        
#         Args:
#             message: User's current message
#             conv_id: Conversation ID for session tracking
#             history: Conversation history from Kotaemon
#             **kwargs: Additional context (e.g., from RAG, file_ids, etc.)
#         """
#         try:
#             # Determine if this is the first turn of Socratic dialogue
#             session = self._get_or_create_session(conv_id)
#             is_first = not session.get("initialized", False)
            
#             # Build initial state
#             initial_state = {
#                 "student_reply": message,
#                 "initial_question": message if is_first else "",
#                 "conversation_id": conv_id,
#                 "is_first_turn": is_first,
#                 "ground_truth_context": kwargs.get("context", ""),
#                 "turn_number": session.get("orchestrator").turn_number if not is_first else 1,
#                 "dialogue_history": session.get("orchestrator").dialogue_history if not is_first else [],
#                 "explored_concepts": session.get("orchestrator").explored_concepts if not is_first else [],
#             }
            
#             # Pass kwargs to ReAct pipeline (file_ids, settings, etc.)
#             if is_first:
#                 self._current_kwargs = kwargs
            
#             # Run the graph
#             result = await self.graph.ainvoke(initial_state)
            
#             # Get final answer
#             final_answer = result.get("final_answer", "Let's continue exploring this topic.")
            
#             return Document(content=final_answer, channel="chat")
            
#         except Exception as e:
#             logger.error(f"Error in Socratic pipeline ainvoke: {e}", exc_info=True)
#             return Document(
#                 content=f"I encountered an error: {str(e)}\n\nPlease try again or rephrase your question.",
#                 channel="chat"
#             )

#     @classmethod
#     def get_pipeline(
#         cls,
#         settings: dict,
#         states: dict,
#         retrievers: list | None = None,
#     ) -> BaseReasoning:
#         """
#         Factory method to create pipeline instance with user settings.
#         Called by Kotaemon to instantiate the pipeline.
        
#         Args:
#             settings: User settings from Kotaemon UI
#             states: Application state
#             retrievers: List of retriever components for RAG
            
#         Returns:
#             Configured SocraticPipeline instance
#         """
#         _id = cls.get_info()["id"]
#         prefix = f"reasoning.options.{_id}"
        
#         # Get LLM from settings
#         llm_name = settings.get(f"{prefix}.llm", "")
#         llm = llms.get(llm_name) if llm_name else llms.get_default()
        
#         # Get Socratic-specific settings
#         max_turns = settings.get(f"{prefix}.max_turns", 8)
#         stuck_threshold = settings.get(f"{prefix}.stuck_threshold", 3)
        
#         # Create pipeline instance
#         pipeline = cls(
#             llm=llm,
#             max_turns=max_turns,
#             stuck_threshold=stuck_threshold,
#             retrievers=retrievers or []
#         )
        
#         # Configure the internal ReAct pipeline with retrievers and settings
#         pipeline.react_pipeline = ReactAgentPipeline.get_pipeline(
#             settings=settings,
#             states=states,
#             retrievers=retrievers
#         )
        
#         return pipeline

#     @classmethod
#     def get_info(cls) -> dict:
#         """Kotaemon plugin metadata"""
#         return {
#             "id": "socratic_tutor",
#             "name": "Socratic Tutor",
#             "description": (
#                 "An interactive Socratic tutoring system that guides learners through "
#                 "probing questions, evaluates their reasoning, and provides reflective feedback. "
#                 "Uses RAG to ground dialogue in your uploaded notes."
#             ),
#         }

#     @classmethod
#     def get_user_settings(cls) -> dict:
#         """Kotaemon user-configurable settings"""
#         llm_choices = [("(default)", "")]
#         try:
#             llm_choices += [(k, k) for k in llms.options().keys()]
#         except Exception as e:
#             logger.exception(f"Failed to get LLM options: {e}")
            
#         return {
#             "llm": {
#                 "name": "Language Model",
#                 "value": "",
#                 "component": "dropdown",
#                 "choices": llm_choices,
#                 "special_type": "llm",
#                 "info": "The LLM to drive the Socratic tutor. If None, the application default will be used.",
#             },
#             "max_turns": {
#                 "name": "Maximum Turns",
#                 "value": 8,
#                 "component": "number",
#                 "info": "Maximum number of Socratic exchanges before forcing reflection.",
#             },
#             "stuck_threshold": {
#                 "name": "Stuck Threshold",
#                 "value": 3,
#                 "component": "number",
#                 "info": "Number of confused responses before offering a hint.",
#             },
#         }