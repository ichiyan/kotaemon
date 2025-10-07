import logging
from typing import Optional, Union
import json
import re

from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM, PromptTemplate
from .evaluator_prompt import socratic_evaluator_prompt


logger = logging.getLogger(__name__)

class SocraticEvaluatorAgent(BaseAgent):
    """
    Evaluator that provides detailed assessment of student understanding and recommends next action.
    """

    name: str = "SocraticEvaluatorAgent"
    agent_type: AgentType = AgentType.socratic_evaluator
    description: str = "Evaluates student responses and recommends dialogue flow"

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

        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(template=prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            self.prompt_template = prompt_template
        else:
            self.prompt_template = socratic_evaluator_prompt


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
        Evaluate student response.
        
        Args:
            instruction: dict with:
                - context: ground truth
                - history: conversation history
        
        Returns:
            AgentOutput with decision and detailed metadata
        """
        try:
            context = instruction.get("context", "")
            history = instruction.get("history", [])
            turn_number = instruction.get("turn_number", 1)
            max_turns = instruction.get("max_turns", 5)

            
            prompt = self.prompt_template.populate(
                context=context,
                history=self._format_history(history), 
                turn_number=turn_number, 
                max_turns=max_turns
            )

            result = self.llm(prompt)

            output_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

            # Try to parse JSON output from the model
            try:
                parsed = json.loads(output_text)
                decision = parsed.get("decision", "continue")
            except json.JSONDecodeError:
                # fallback heuristic
                if "reflect" in output_text.lower():
                    decision = "reflect"
                elif "hint" in output_text.lower():
                    decision = "hint"
                else:
                    decision = "continue_socratic"
                parsed = {
                    "decision": decision,
                    "understanding_level": "unknown",
                    "reasoning": output_text,
                    "student_analysis": "",
                    "key_points_understood": [],
                    "gaps_identified": [],
                }

            
            return AgentOutput(
                text=decision,
                agent_type=self.agent_type,
                status="finished",
                metadata=parsed,
            )
            
        except Exception as e:
            return AgentOutput(
                text="continue",  # Safe fallback
                agent_type=self.agent_type,
                status="failed",
                error=str(e),
            )

