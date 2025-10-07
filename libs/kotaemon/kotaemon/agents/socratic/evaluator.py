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

    # def _parse_evaluation(self, output_text: str) -> dict:
    #     """Parse LLM output into structured evaluation"""
    #     # Try to extract JSON
    #     try:
    #         # Find JSON object in the response
    #         json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    #         if json_match:
    #             eval_dict = json.loads(json_match.group())
    #             return eval_dict
    #     except json.JSONDecodeError as e:
    #         logger.error(f"An error occurred while parsing evaluation output: {str(e)}")
        
    #     # Fallback: extract decision from text
    #     output_lower = output_text.lower()
    #     decision = "continue_socratic"  # default
        
    #     if "hint" in output_lower and "decision" in output_lower:
    #         decision = "hint"
    #     elif "reflect" in output_lower and "decision" in output_lower:
    #         decision = "reflect"
    #     elif "continue" in output_lower:
    #         decision = "continue_socratic"
        
    #     # Infer understanding level
    #     understanding = "partial"
    #     if any(word in output_lower for word in ["confused", "stuck", "unclear"]):
    #         understanding = "confused"
    #     elif any(word in output_lower for word in ["strong", "excellent", "comprehensive", "well"]):
    #         understanding = "strong"
        
    #     return {
    #         "decision": decision,
    #         "understanding_level": understanding,
    #         "reasoning": output_text[:200],  # First 200 chars as reasoning
    #         "student_analysis": output_text,
    #         "key_points_understood": [],
    #         "gaps_identified": []
    #     }

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

            
            history_str = self._format_history(history)
            
            prompt = self.prompt_template.populate(
                context=context,
                history=self._format_history(history)
            )

            result = self.llm(prompt)
            output_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

            
            return AgentOutput(
                text=output_text,
                agent_type=self.agent_type,
                status="finished",
                metadata={}
            )
            
        except Exception as e:
            return AgentOutput(
                text="continue",  # Safe fallback
                agent_type=self.agent_type,
                status="failed",
                error=str(e),
            )


    # def run(self, instruction: dict) -> AgentOutput:
    #     """
    #     Evaluate student response.
        
    #     Args:
    #         instruction: dict with:
    #             - context: ground truth
    #             - student_reply: current student message
    #             - history: conversation history
    #             - turn_number: current turn
    #             - max_turns: maximum turns allowed
        
    #     Returns:
    #         AgentOutput with decision and detailed metadata
    #     """
    #     try:
    #         context = instruction.get("context", "")
    #         student_reply = instruction.get("student_reply", "")
    #         history = instruction.get("history", [])
    #         turn_number = instruction.get("turn_number", 1)
    #         max_turns = instruction.get("max_turns", 8)

    #         history_str = self._format_history(history)
            
    #         prompt = self.prompt_template.populate(
    #             context=context,
    #             student_reply=student_reply,
    #             history=history_str,
    #             turn_number=turn_number,
    #             max_turns=max_turns,
    #         )

    #         result = self.llm(prompt)
    #         output_text = result.text.strip() if hasattr(result, "text") else str(result).strip()

    #         # Parse evaluation
    #         evaluation = self._parse_evaluation(output_text)
            
    #         # Validate decision
    #         valid_decisions = {"continue_socratic", "hint", "reflect"}
    #         decision = evaluation.get("decision", "continue_socratic")
    #         if decision not in valid_decisions:
    #             decision = "continue_socratic"
            
    #         return AgentOutput(
    #             text=decision,
    #             agent_type=self.agent_type,
    #             status="finished",
    #             metadata={
    #                 "understanding_level": evaluation.get("understanding_level", "partial"),
    #                 "reasoning": evaluation.get("reasoning", ""),
    #                 "student_analysis": evaluation.get("student_analysis", ""),
    #                 "key_points_understood": evaluation.get("key_points_understood", []),
    #                 "gaps_identified": evaluation.get("gaps_identified", []),
    #                 "full_evaluation": evaluation,
    #             }
    #         )
            
    #     except Exception as e:
    #         return AgentOutput(
    #             text="continue_socratic",  # Safe fallback
    #             agent_type=self.agent_type,
    #             status="failed",
    #             error=str(e),
    #         )