from typing import Optional, Union, List, Dict
from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM, PromptTemplate
from .questioner_prompt import socratic_prompt, hint_generation_prompt, synthesis_prompt


class SocraticQuestionerAgent(BaseAgent):
    """
    Generates probing/guiding questions based on context and student state
    """

    name: str = "SocraticQuestionerAgent"
    agent_type: AgentType = AgentType.socratic_questioner
    description: str = (
        "Generates Socratic questions to guide student learning"
    )

    class Config:
        allow_extra = True

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[Union[PromptTemplate, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm or ChatLLM.default()
       

        # Configure prompts
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(template=prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            self.prompt_template = prompt_template
        else:
            self.prompt_template = socratic_prompt

        # self.hint_prompt = hint_generation_prompt
        # self.synthesis_prompt = synthesis_prompt

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No prior conversation."
        
        formatted = []
        for entry in history:
            for role, msg in entry.items():  #only one key-value pair per list item
                formatted.append(f"{role}: {msg}")
        
        return "\n".join(formatted)
    

    def run(self, instruction: dict) -> AgentOutput:
        """
        Generate Socratic questions.
        
        Args:
            instruction: dict with:
                - context: ground truth from ReAct agent
                - history: list of dictionaries e.g. [{"agent": "message"}]
        
        Returns:
            AgentOutput with questions and metadata
        """
        try:
            # Extract inputs
            context = instruction.get("context", "")
            history = instruction.get("history", [])
          

            prompt = self.prompt_template.populate(
                context=context,
                history=self._format_history(history)
            )

            # Generate questions
            result = self.llm(prompt)
            output_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

            # Return with metadata for orchestrator
            return AgentOutput(
                text=output_text,
                agent_type=self.agent_type,
                status="finished",
                metadata={}
            )

        except Exception as e:
            return AgentOutput(
                text="",
                agent_type=self.agent_type,
                status="failed",
                error=str(e),
            )

    # def _extract_concepts(self, text: str) -> List[str]:
    #     """Extract key concepts mentioned in text"""
    #     import re
    #     # Simple extraction - can be enhanced with NER
    #     words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\w{4,}\b', text)
    #     return list(set(words))[:5]


    # def run(self, instruction: dict) -> AgentOutput:
    #     """
    #     Generate Socratic questions.
        
    #     Args:
    #         instruction: dict with:
    #             - context: ground truth from ReAct agent
    #             - student_reply: student's current message
    #             - history: list of {"role": str, "content": str}
    #             - turn_number: current turn
    #             - max_turns: max allowed turns
    #             - explored_concepts: list of concepts already discussed
    #             - understanding_level: "confused" | "partial" | "strong"
    #             - student_analysis: evaluator's analysis (optional)
    #             - questioning_strategy: "standard" | "hint" | "synthesis"
        
    #     Returns:
    #         AgentOutput with questions and metadata
    #     """
    #     try:
    #         # Extract inputs
    #         context = instruction.get("context", "")
    #         student_reply = instruction.get("student_reply", "")
    #         history = instruction.get("history", [])
    #         turn_number = instruction.get("turn_number", 1)
    #         max_turns = instruction.get("max_turns", 8)
    #         explored_concepts = instruction.get("explored_concepts", [])
    #         understanding_level = instruction.get("understanding_level", "partial")
    #         student_analysis = instruction.get("student_analysis", "")
    #         strategy = instruction.get("questioning_strategy", "standard")
    #         stuck_threshold = instruction.get("stuck_threshold", 3)
            
    #         # Format inputs
    #         history_str = self._format_history(history)
    #         explored_str = ", ".join(explored_concepts) if explored_concepts else "None yet"
            
    #         # Update explored concepts with current reply
    #         new_concepts = self._extract_concepts(student_reply)
    #         all_explored = list(set(explored_concepts + new_concepts))
            
    #         # Select appropriate prompt based on strategy
    #         if strategy == "hint":
    #             prompt = self.hint_prompt.populate(
    #                 context=context,
    #                 student_reply=student_reply,
    #                 history=history_str
    #             )
    #         elif strategy == "synthesis":
    #             prompt = self.synthesis_prompt.populate(
    #                 context=context,
    #                 student_reply=student_reply,
    #                 explored_concepts=explored_str
    #             )
    #         else:  # standard
    #             prompt = self.prompt_template.populate(
    #                 context=context,
    #                 student_reply=student_reply,
    #                 history=history_str,
    #                 turn_number=turn_number,
    #                 max_turns=max_turns,
    #                 explored_concepts=explored_str,
    #                 understanding_level=understanding_level,
    #                 student_analysis=student_analysis or "No prior analysis.",
    #                 stuck_threshold=stuck_threshold
    #             )

    #         # Generate questions
    #         result = self.llm(prompt)
    #         output_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

    #         # Return with metadata for orchestrator
    #         return AgentOutput(
    #             text=output_text,
    #             agent_type=self.agent_type,
    #             status="finished",
    #             metadata={
    #                 "explored_concepts": all_explored,
    #                 "understanding_level": understanding_level,
    #                 "strategy_used": strategy,
    #                 "turn_number": turn_number,
    #             }
    #         )

    #     except Exception as e:
    #         return AgentOutput(
    #             text="",
    #             agent_type=self.agent_type,
    #             status="failed",
    #             error=str(e),
    #         )