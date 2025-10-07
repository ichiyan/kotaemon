from typing import Optional, Union, List, Dict, Any
from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM, PromptTemplate
# from .reflection_prompt import socratic_reflection_prompt, metacognitive_reflection_prompt
from .reflection_prompt import socratic_reflection_prompt


reflection_fallback = """## Reflection

Thank you for engaging in this Socratic dialogue! While I encountered an 
error generating a detailed reflection, I want to acknowledge your effort 
in thinking through these concepts.

## 🚀 Keep Learning

Continue to:
- Ask questions when you're curious
- Challenge your own assumptions
- Connect new ideas to what you already know
- Practice explaining concepts in your own words

Feel free to ask more questions anytime!"""


class SocraticReflectionAgent(BaseAgent):
    """
    Provides comprehensive reflection at the end of Socratic dialogue.
    Analyzes the learning journey, assesses understanding, and provides
    encouragement and next steps.
    """

    name: str = "SocraticReflectionAgent"
    agent_type: AgentType = AgentType.socratic_reflection
    description: str = "Provides comprehensive learning reflection and summary"

    class Config:
        allow_extra = True

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        prompt_template: Optional[Union[PromptTemplate, str]] = None,
        include_metacognitive_analysis: bool = False,
        **kwargs,
    ):
        """
        Args:
            llm: Language model for generating reflections
            prompt_template: Custom prompt (optional)
            include_metacognitive_analysis: Whether to include metacognitive analysis
        """
        super().__init__(**kwargs)
        self.llm = llm or ChatLLM.default()
        self.include_metacognitive = include_metacognitive_analysis

        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(template=prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            self.prompt_template = prompt_template
        else:
            self.prompt_template = socratic_reflection_prompt

        # self.metacognitive_prompt = metacognitive_reflection_prompt

    
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
        Generate reflection on the Socratic dialogue.
        
        Args:
            instruction: dict with:
                - history: conversation history (list or string)
                - context: ground truth from ReAct agent
        
        Returns:
            AgentOutput with comprehensive reflection
        """
        try:
            user_query = instruction.get("user_query", "")
            context = instruction.get("context", "")
            history = instruction.get("history", [])
            eval_decision = instruction.get("latest_eval_decision", "continue")
            evaluation = instruction.get("latest_eval", {})

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
          

            # Generate main reflection
            prompt = self.prompt_template.populate(
                context=context,
                history=self._format_history(history),
                evaluation_assessment=evaluation_assessment
            )
            
            result = self.llm(prompt)
            reflection = result.text.strip() if hasattr(result, "text") else str(result).strip()
            
    
            return AgentOutput(
                text=reflection,
                agent_type=self.agent_type,
                status="finished",
                metadata={}
            )

        except Exception as e:
            # Provide a graceful fallback reflection
            return AgentOutput(
                text=reflection_fallback,
                agent_type=self.agent_type,
                status="finished",
                error=str(e),
                metadata={"is_fallback": True}
            )