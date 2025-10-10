from typing import Optional, Union, List, Dict
from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM, PromptTemplate
from .questioner_prompt import socratic_prompt, initial_socratic_prompt


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
        initial_prompt_template: Optional[Union[PromptTemplate, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm or ChatLLM.default()
       

        # Configure prompts

        if isinstance(initial_prompt_template, str):
            self.initial_prompt_template = PromptTemplate(template=initial_prompt_template)
        elif isinstance(initial_prompt_template, PromptTemplate):
            self.initial_prompt_template = initial_prompt_template
        else:
            self.initial_prompt_template = initial_socratic_prompt


        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(template=prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            self.prompt_template = prompt_template
        else:
            self.prompt_template = socratic_prompt

       
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
           
            user_query = instruction.get("user_query", "")
            context = instruction.get("context", "")
            history = instruction.get("history", [])
            eval_decision = instruction.get("latest_eval_decision", "continue")
            evaluation = instruction.get("latest_eval", {})
            turn_number = instruction.get("turn_number", 0)
            max_turns = instruction.get("max_turns", 5)

            if turn_number == 0:
                prompt = self.initial_prompt_template.populate(
                    user_query=user_query,
                    context=context
                )
            else:
                
                evaluation_assessment = ""

                if eval_decision:
                    evaluation_assessment += f"Decision: {eval_decision}\n"

                if evaluation:
                    if evaluation.get("understanding_level"):
                        evaluation_assessment += f"Understanding Level: {evaluation.get('understanding_level')}\n"
                    
                    if evaluation.get("reasoning"):
                        evaluation_assessment += f"Evaluator's Reasoning: {evaluation.get('reasoning')}\n"
                    
                    if evaluation.get("key_points_understood"):
                        evaluation_assessment += f"Key Points Understood: {evaluation.get('key_points_understood')}\n"
                    
                    if evaluation.get("gaps_identified"):
                        evaluation_assessment += f"Gaps Identified: {evaluation.get('gaps_identified')}"

    
                prompt = self.prompt_template.populate(
                    context=context,
                    history=self._format_history(history), 
                    evaluation_assessment=evaluation_assessment, 
                    turn_number=turn_number, 
                    max_turns=max_turns
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