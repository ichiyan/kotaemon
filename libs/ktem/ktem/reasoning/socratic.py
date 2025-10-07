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
    max_turns: int
    latest_eval: dict[str, Any]
    latest_eval_decision: str
    reflection: str
    mode: str  # "greeting", "user_query", "auto_query", "dialogue"
    greeting_shown: bool

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
            max_turns: int = 5):
        super().__init__()
        self.react_pipeline = react_pipeline
        self.questioner_agent = questioner_agent
        self.evaluator_agent = evaluator_agent
        self.reflection_agent = reflection_agent
        self.max_turns = max_turns
        self.checkpointer = checkpointer
        self.graph = self._build_graph(self.checkpointer)
    
    def _build_graph(self, checkpointer: Checkpointer):
        def show_greeting(state: SocraticState) -> SocraticState:
            """Show initial greeting and ask user preference"""
            greeting = (
                "Hello! I'm your Socratic learning assistant. 🎓\n\n"
                "I can help you explore topics through thoughtful dialogue. "
                "Would you like to:\n\n"
                "1. **Provide your own question** - Type your question and I'll guide you through it\n"
                "2. **Let me generate a question** - Type 'auto' and I'll create a question based on the documents\n\n"
                "What would you prefer?"
            )
            state["socratic_query"] = greeting
            state["greeting_shown"] = True
            state["mode"] = "greeting"
            return state
        
        def route_after_greeting(state: SocraticState):
            """Route based on user's choice after greeting"""
            user_msg = state.get("user_response", "").strip().lower()
            
            if user_msg in ["auto", "2", "generate", "let me generate"]:
                return "generate_auto_query"
            else:
                return "contextualize"
        
        def generate_auto_query(state: SocraticState) -> SocraticState:
            """Generate query based on document summary"""
            print("Generating auto query from documents...")
            
            # Use ReactAgent to summarize documents
            summary_parts = []
            summary_query = "Provide a comprehensive summary of the key concepts and topics covered in the documents."
            
            for response in self.react_pipeline.stream(
                message=summary_query,
                conv_id=f"{state['conv_id']}_summary",
                history=[]
            ):
                if isinstance(response, Document) and response.channel == "chat":
                    if response.content:
                        summary_parts.append(response.content)
            
            doc_summary = "".join(summary_parts)
            
            # Generate a thoughtful question based on the summary
            query_prompt = f"""Based on this document summary:
{doc_summary}

Generate ONE thought-provoking question that would help a student explore and understand the most important concept from these documents. The question should:
- Focus on a core concept or principle
- Encourage critical thinking
- Be clear and specific
- Be suitable for Socratic dialogue

Return ONLY the question, nothing else."""
            
            # Use the questioner agent's LLM to generate the question
            generated_query = self.questioner_agent.llm.invoke(query_prompt).text
            
            state["user_query"] = generated_query
            state["context"] = doc_summary  # Use summary as context
            state["mode"] = "auto_query"
            
            print(f"Generated query: {generated_query}")
            return state
        
        def decide_next_node(state: SocraticState):
            print(f"DEBUG Router - Turn {state['turn_number']}/{state['max_turns']}")
            
            if state["turn_number"] > self.max_turns:
                return "reflect"
            
            next_action = state.get("latest_eval_decision", "continue")
            if next_action not in ["continue", "hint", "reflect"]:
                next_action = "continue"
            print(f"NEXT ACTION: {next_action}")
            return next_action
        
        def generate_context(state: SocraticState) -> SocraticState:
            """Generate context from user's query using RAG"""
            # Only generate context if not already set (from auto mode)
            if state.get("mode") == "auto_query" and state.get("context"):
                print("Skipping contextualize - using existing context from auto query")
                return state
            
            print(f"Generating context for query: {state['user_query']}")
            context_parts = []
            
            for response in self.react_pipeline.stream(
                message=state["user_query"],
                conv_id=f"{state['conv_id']}_context",
                history=[]
            ):
                if isinstance(response, Document) and response.channel == "chat":
                    if response.content:
                        context_parts.append(response.content)
            
            state["context"] = "".join(context_parts)
            state["mode"] = "user_query"
            return state
        
        def generate_socratic_query(state: SocraticState) -> SocraticState:
            """Generate Socratic question"""
            query = self.questioner_agent({
                "user_query": state.get("user_query", ""),
                "context": state.get("context", ""), 
                "history": state.get("history", []), 
                "latest_eval_decision": state.get("latest_eval_decision", "continue"),
                "latest_eval": state.get("latest_eval", {}), 
                "turn_number": state.get("turn_number", 0), 
                "max_turns": state.get("max_turns", self.max_turns)
            })
            state["socratic_query"] = query.text
            state["history"].append({self.questioner_agent.agent_type: query.text})
            state["turn_number"] += 1
            state["mode"] = "dialogue"
            return state 
        
        def get_user_response(state: SocraticState) -> SocraticState:
            """Get user response via interrupt"""
            user_response = interrupt(value="")
            state["user_response"] = user_response
            state["history"].append({"user": user_response})
            print(f"GET_USER_RESPONSE: {user_response}")
            return state
        
        def evaluate_user_response(state: SocraticState) -> SocraticState:
            """Evaluate user's response"""
            print("EVALUATING")
            result = self.evaluator_agent({  
                "context": state.get("context", ""), 
                "history": state.get("history", []), 
                "turn_number": state.get("turn_number", 0), 
                "max_turns": state.get("max_turns", self.max_turns)
            })
            state["latest_eval_decision"] = result.text
            state["latest_eval"] = result.metadata
            return state
        
        def generate_reflection(state: SocraticState) -> SocraticState:
            """Generate final reflection"""
            state["reflection"] = self.reflection_agent({
                "user_query": state.get("user_query", ""),
                "context": state.get("context", ""), 
                "history": state.get("history", []), 
                "latest_eval_decision": state.get("latest_eval_decision", "continue"),
                "latest_eval": state.get("latest_eval", {}), 
            }).text
            return state
        
        # Build the graph
        graph = StateGraph(SocraticState)
        
        # Add nodes
        graph.add_node("greeting", show_greeting)
        graph.add_node("get_user_choice", get_user_response)
        graph.add_node("generate_auto_query", generate_auto_query)
        graph.add_node("contextualize", generate_context)
        graph.add_node("question", generate_socratic_query)
        graph.add_node("user_response", get_user_response)
        graph.add_node("evaluate", evaluate_user_response)
        graph.add_node("reflect", generate_reflection)
        
        # Define edges
        graph.add_edge(START, "greeting")
        graph.add_edge("greeting", "get_user_choice")
        
        # Route after user chooses
        graph.add_conditional_edges(
            "get_user_choice",
            route_after_greeting,
            {
                "generate_auto_query": "generate_auto_query",
                "contextualize": "contextualize"
            }
        )
        
        # Both paths converge to question
        graph.add_edge("generate_auto_query", "question")
        graph.add_edge("contextualize", "question")
        
        # Dialogue loop
        graph.add_edge("question", "user_response")
        graph.add_edge("user_response", "evaluate")
        graph.add_conditional_edges(
            "evaluate", 
            decide_next_node, 
            { 
                "continue": "question", 
                "hint": "question",
                "reflect": "reflect"
            }
        )
        graph.add_edge("reflect", END)
        
        compiled_graph = graph.compile(checkpointer=checkpointer)
        return compiled_graph
    
    def stream(self, message, conv_id: str, history: list, pipeline_state: dict = None, **kwargs):
        """Stream method for SocraticPipeline"""
        if self.graph is None:
            raise RuntimeError("Pipeline not properly initialized. Use get_pipeline() classmethod.")
        
        config = {"configurable": {"thread_id": conv_id}}
        
        # Check if we're resuming by looking at the actual graph state
        is_resuming = False
        try:
            current_state = self.graph.get_state(config)
            is_resuming = (
                current_state 
                and current_state.values 
                and current_state.values.get("greeting_shown")  # Has been initialized
                and current_state.next  # Has pending nodes (interrupted)
            )
            print(f"DEBUG stream: Checkpoint exists: {current_state and current_state.values is not None}")
            print(f"DEBUG stream: Greeting shown: {current_state.values.get('greeting_shown') if current_state.values else 'N/A'}")
            print(f"DEBUG stream: Next nodes: {current_state.next if current_state else 'N/A'}")
            print(f"DEBUG stream: is_resuming={is_resuming}")
        except Exception as e:
            print(f"DEBUG stream: Error checking state: {e}")
            is_resuming = False
        
        if is_resuming:
            # Resume from interrupt with user's response
            print(f"DEBUG stream: Resuming with message: {message}")
            try:
                for chunk in self.graph.stream(Command(resume=message), config, stream_mode="updates"):
                    yield from self._process_chunk(chunk)
            except Exception as e:
                print(f"ERROR during resume: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Start new conversation with greeting
            print(f"DEBUG stream: Starting new conversation")
            initial_state = SocraticState(
                conv_id=conv_id,
                context="", 
                user_query="",  # Will be filled later
                socratic_query="",
                user_response="", 
                history=[], 
                turn_number=0, 
                max_turns=self.max_turns,
                latest_eval={},
                latest_eval_decision="continue",
                reflection="",
                mode="greeting",
                greeting_shown=False
            )
            
            try:
                for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
                    yield from self._process_chunk(chunk)
            except Exception as e:
                print(f"ERROR during start: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_chunk(self, chunk):
        """Helper method to process graph chunks"""
        for node_id, value in chunk.items():
            if node_id == "greeting":
                yield Document(
                    channel="chat",
                    content=value["socratic_query"],
                )
            elif node_id == "question":
                # Add context info for auto-generated queries
                query_text = value["socratic_query"]
                if value.get("mode") == "auto_query" and value.get("turn_number") == 1:
                    query_text = f"**Generated Question:** {value['user_query']}\n\n{query_text}"
                yield Document(
                    channel="chat",
                    content=query_text,
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
            elif node_id == "evaluate":
                print(f"[Evaluator Decision] → {value['latest_eval_decision']}")
                print(f"[Evaluation Metadata]: {value['latest_eval']}")
    
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

# class SocraticState(TypedDict):
#     conv_id: str
#     context: str
#     user_query: str  
#     socratic_query: str 
#     user_response: str
#     history: list[dict[str, str]]
#     turn_number: int 
#     max_turns: int
#     latest_eval: dict[str, Any]
#     latest_eval_decision: str
#     reflection: str
    


# class SocraticPipeline(BaseReasoning):

#     class Config:
#         allow_extra = True
 

#     def __init__(
#             self, 
#             react_pipeline: ReactAgentPipeline = ReactAgentPipeline.withx(), 
#             questioner_agent: SocraticQuestionerAgent = SocraticQuestionerAgent.withx(), 
#             evaluator_agent: SocraticEvaluatorAgent = SocraticEvaluatorAgent.withx(), 
#             reflection_agent: SocraticReflectionAgent = SocraticReflectionAgent.withx(), 
#             checkpointer: Checkpointer = MemorySaver(), 
#             max_turns: int =5):
#         super().__init__()
#         self.react_pipeline = react_pipeline
#         self.questioner_agent = questioner_agent
#         self.evaluator_agent = evaluator_agent
#         self.reflection_agent = reflection_agent
#         self.max_turns = max_turns
#         self.checkpointer = checkpointer
#         self.graph = self._build_graph(self.checkpointer)



#     def _build_graph(self, checkpointer: Checkpointer):

#         def decide_next_node(state: SocraticState):
            
#             print(f"DEBUG Router - {state}")

#             if state["turn_number"] > self.max_turns:
#                 return "reflect"
            
#             next_action = state.get("latest_eval_decision", "continue")

#             if next_action not in ["continue", "hint", "reflect"]:
#                 next_action = "continue"

#             print(f"NEXT ACTION: {next_action}")
#             return next_action



#         def generate_context(state: SocraticState) -> SocraticState:
#             context_parts = []
            
#             for response in self.react_pipeline.stream(
#                 message=state["user_query"],
#                 conv_id="",
#                 history=[]
#             ):
#                 if isinstance(response, Document) and response.channel == "chat":
#                     if response.content:
#                         context_parts.append(response.content)
            
        
#             state["context"] = "".join(context_parts)

#             return state
        
        
#         def generate_socratic_query(state: SocraticState) -> SocraticState:
#             query = self.questioner_agent({
#                 "user_query": state.get("user_query", ""),
#                 "context": state.get("context", ""), 
#                 "history": state.get("history", []), 
#                 "latest_eval_decision": state.get("latest_eval_decision", "continue"),
#                 "latest_eval": state.get("latest_eval", {}), 
#                 "turn_number": state.get("turn_number", 0), 
#                 "max_turns": state.get("max_turns", self.max_turns)
#             })

#             state ["socratic_query"] = query.text
#             state["history"].append({self.questioner_agent.agent_type: query.text})
#             state["turn_number"] += 1

#             return state 

        
#         def get_user_response(state: SocraticState) -> SocraticState:
#             user_response = interrupt(value="")
#             state["user_response"] = user_response
#             state["history"].append({"user": user_response})

#             print(f"GET_USER_RESPONSE: {state}")

#             return state
                
        

#         def evaluate_user_response(state: SocraticState) -> SocraticState:
#             print("EVALUATING")
#             result = self.evaluator_agent({  
#                 "context": state.get("context", ""), 
#                 "history": state.get("history", []), 
#                 "turn_number": state.get("turn_number", 0), 
#                 "max_turns": state.get("max_turns", self.max_turns)
#             })

#             state["latest_eval_decision"] = result.text
#             state["latest_eval"] = result.metadata

#             return state
        

#         def generate_reflection(state: SocraticState) -> SocraticState:
#             state["reflection"] = self.reflection_agent({
#                 "user_query": state.get("user_query", ""),
#                 "context": state.get("context", ""), 
#                 "history": state.get("history", []), 
#                 "latest_eval_decision": state.get("latest_eval_decision", "continue"),
#                 "latest_eval": state.get("latest_eval", {}), 
#             }).text


#             return state


#         graph = StateGraph(SocraticState)

       
#         graph.add_node("contextualize", generate_context)
#         graph.add_node("question", generate_socratic_query)
#         graph.add_node("user_response", get_user_response)
#         graph.add_node("evaluate", evaluate_user_response)
#         graph.add_node("reflect", generate_reflection)

#         graph.add_edge(START, "contextualize")
#         graph.add_edge("contextualize", "question")
#         graph.add_edge("question", "user_response")
#         graph.add_edge("user_response", "evaluate")

#         graph.add_conditional_edges(
#             "evaluate", 
#             decide_next_node, 
#             { 
#                 "continue": "question", 
#                 "hint": "question",
#                 "reflect": "reflect"
#             }
#         )

#         graph.add_edge("reflect", END)
        

#         compiled_graph = graph.compile(checkpointer=checkpointer)

#         return compiled_graph
    

#     def stream(self, message, conv_id: str, history: list, pipeline_state: dict = None, **kwargs):
#         """Stream method for SocraticPipeline"""
#         if self.graph is None:
#             raise RuntimeError("Pipeline not properly initialized. Use get_pipeline() classmethod.")
        
#         config = {"configurable": {"thread_id": conv_id}}
        
#         # Check if we're resuming by looking at the actual graph state
#         is_resuming = False
#         try:
#             current_state = self.graph.get_state(config)
#             # We're resuming if:
#             # 1. There's an existing checkpoint with values
#             # 2. The graph is in an interrupted state (next is not empty)
#             is_resuming = (
#                 current_state 
#                 and current_state.values 
#                 and current_state.values.get("context")  # Has been initialized
#                 and current_state.next  # Has pending nodes (interrupted)
#             )
#             print(f"DEBUG stream: Checkpoint exists: {current_state and current_state.values is not None}")
#             print(f"DEBUG stream: Has context: {current_state.values.get('context') if current_state.values else 'N/A'}")
#             print(f"DEBUG stream: Next nodes: {current_state.next if current_state else 'N/A'}")
#             print(f"DEBUG stream: is_resuming={is_resuming}")
#         except Exception as e:
#             print(f"DEBUG stream: Error checking state: {e}")
#             is_resuming = False
        
#         if is_resuming:
#             # Resume from interrupt with user's response
#             print(f"DEBUG stream: Resuming with message: {message}")
#             try:
#                 for chunk in self.graph.stream(Command(resume=message), config, stream_mode="updates"):
#                     yield from self._process_chunk(chunk)
#             except Exception as e:
#                 print(f"ERROR during resume: {e}")
#                 import traceback
#                 traceback.print_exc()
#         else:
#             # Start new conversation
#             print(f"DEBUG stream: Starting new conversation with message: {message}")
#             initial_state = SocraticState(
#                 conv_id=conv_id,
#                 context="", 
#                 user_query=message, 
#                 socratic_query="",
#                 user_response="", 
#                 history=[], 
#                 turn_number=0, 
#                 max_turns=self.max_turns,
#                 latest_eval={},
#                 latest_eval_decision="continue",
#                 reflection=""
#             )
            
#             try:
#                 for chunk in self.graph.stream(initial_state, config, stream_mode="updates"):
#                     yield from self._process_chunk(chunk)
#             except Exception as e:
#                 print(f"ERROR during start: {e}")
#                 import traceback
#                 traceback.print_exc()



#     def _process_chunk(self, chunk):
#         """Helper method to process graph chunks"""
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
#                 yield Document(
#                     channel="interrupt",
#                     content={"waiting_for_input": True}
#                 )
#             elif node_id == "evaluate":
#                 print(f"[Evaluator Decision] → {value['latest_eval_decision']}")
#                 print(f"[Evaluation Metadata]: {value['latest_eval']}")


    
#     @classmethod
#     def get_pipeline(
#         cls, settings: dict, states: dict, retrievers: list | None = None
#     ) -> BaseReasoning:
#         _id = cls.get_info()["id"]
#         prefix = f"reasoning.options.{_id}"

#         llm_name = settings[f"{prefix}.llm"]
#         llm = llms.get(llm_name, llms.get_default())
#         max_turns = settings[f"{prefix}.dialogue_max_turns"]

#         react_pipeline = ReactAgentPipeline.get_pipeline(settings=settings, states=states, retrievers=retrievers)
        
#         pipeline = SocraticPipeline(
#             react_pipeline=react_pipeline, 
#             questioner_agent=SocraticQuestionerAgent(llm=llm), 
#             evaluator_agent=SocraticEvaluatorAgent(llm=llm), 
#             reflection_agent=SocraticReflectionAgent(llm=llm), 
#             checkpointer=MemorySaver(),
#             max_turns=max_turns
#         )

#         return pipeline

#     @classmethod
#     def get_user_settings(cls) -> dict:
#         settings = ReactAgentPipeline.get_user_settings()
#         settings.update(
#             {
#                 "dialogue_max_turns": {
#                     "name": "Maximum Socratic dialogue turns",
#                     "value": 5,
#                     "component": "number",
#                     "info": "Maximum number of Socratic dialogue exchanges before reflection.",
#                 }
#             }
#         )

#         return settings
    

#     @classmethod
#     def get_info(cls) -> dict:
#         return {
#             "id": "Socratic",
#             "name": "Socratic",
#             "description": (
#                 "Combines a ReAct agent (for factual grounding) with an adaptive "
#                 "Socratic dialogue loop that asks probing questions and evaluates responses."
#             ),
#         }
        


