from typing import Optional, Union, List, Dict, Any
from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM, PromptTemplate
from .reflection_prompt import socratic_reflection_prompt, metacognitive_reflection_prompt


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

        self.metacognitive_prompt = metacognitive_reflection_prompt

    def _format_history(self, history: Union[List[Dict], List[str], str]) -> str:
        """Format conversation history for reflection"""
        if isinstance(history, str):
            return history
        
        if isinstance(history, list):
            if not history:
                return "No conversation history available."
            
            formatted = []
            for i, entry in enumerate(history, 1):
                if isinstance(entry, dict):
                    role = entry.get("role", "unknown")
                    content = entry.get("content", "")
                    formatted.append(f"[Exchange {i//2 + 1}]")
                    formatted.append(f"{role.capitalize()}: {content}")
                else:
                    formatted.append(str(entry))
            
            return "\n".join(formatted)
        
        return str(history)

    def _extract_explored_concepts(self, history: Union[List, str]) -> List[str]:
        """Extract concepts that were discussed during the dialogue"""
        import re
        
        history_str = self._format_history(history)
        
        # Simple keyword extraction - could be enhanced with NER
        # Look for capitalized words and technical terms
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', history_str)
        words.extend(re.findall(r'\b[a-z]{5,}\b', history_str.lower()))
        
        # Remove common words
        common_words = {
            'student', 'tutor', 'think', 'question', 'answer', 'understand',
            'explain', 'because', 'would', 'could', 'should', 'about', 'there'
        }
        concepts = [w for w in set(words) if w.lower() not in common_words]
        
        return concepts[:8]  # Top 8 concepts

    def _count_turns(self, history: Union[List, str]) -> int:
        """Count the number of dialogue turns"""
        if isinstance(history, list):
            # Each pair of student/tutor is one turn
            return len(history) // 2
        elif isinstance(history, str):
            # Count occurrences of "Student:" or similar markers
            import re
            student_turns = len(re.findall(r'[Ss]tudent:', history))
            return student_turns
        return 0

    def _infer_final_understanding(
        self, 
        history: Union[List, str], 
        metadata: Optional[Dict] = None
    ) -> str:
        """Infer student's final understanding level"""
        if metadata and "understanding_level" in metadata:
            return metadata["understanding_level"]
        
        # Fallback: analyze last student response
        history_str = self._format_history(history)
        last_student_msg = ""
        
        if isinstance(history, list) and history:
            # Get last student message
            for entry in reversed(history):
                if isinstance(entry, dict) and entry.get("role") == "student":
                    last_student_msg = entry.get("content", "").lower()
                    break
        
        if not last_student_msg:
            # Try to extract from string
            import re
            matches = re.findall(r'student:\s*([^\n]+)', history_str.lower())
            if matches:
                last_student_msg = matches[-1]
        
        # Simple heuristic
        confusion_markers = ["don't know", "not sure", "confused", "?"]
        strong_markers = ["understand", "realize", "see now", "because", "therefore"]
        
        if any(marker in last_student_msg for marker in confusion_markers):
            return "confused"
        elif any(marker in last_student_msg for marker in strong_markers):
            return "strong"
        else:
            return "partial"

    def _generate_metacognitive_analysis(
        self,
        history: str,
        context: str
    ) -> str:
        """Generate metacognitive analysis if requested"""
        try:
            prompt = self.metacognitive_prompt.populate(
                history=history,
                context=context
            )
            
            result = self.llm(prompt)
            analysis = result.text.strip() if hasattr(result, "text") else str(result).strip()
            
            return f"\n\n## 🧠 Metacognitive Analysis\n{analysis}"
        except Exception as e:
            return ""

    def run(self, instruction: dict) -> AgentOutput:
        """
        Generate reflection on the Socratic dialogue.
        
        Args:
            instruction: dict with:
                - history: conversation history (list or string)
                - context: ground truth from ReAct agent
                - explored_concepts: list of concepts discussed (optional)
                - final_understanding: student's final understanding level (optional)
                - metadata: additional metadata from orchestrator (optional)
        
        Returns:
            AgentOutput with comprehensive reflection
        """
        try:
            # Extract inputs
            history = instruction.get("history", [])
            context = instruction.get("context", "")
            explored_concepts = instruction.get("explored_concepts", [])
            final_understanding = instruction.get("final_understanding")
            metadata = instruction.get("metadata", {})
            
            # Format history
            history_str = self._format_history(history)
            
            # Extract or use provided concepts
            if not explored_concepts:
                explored_concepts = self._extract_explored_concepts(history)
            concepts_str = ", ".join(explored_concepts) if explored_concepts else "Various topics"
            
            # Count turns
            turn_count = self._count_turns(history)
            
            # Infer understanding if not provided
            if not final_understanding:
                final_understanding = self._infer_final_understanding(history, metadata)
            
            # Generate main reflection
            prompt = self.prompt_template.populate(
                context=context,
                history=history_str,
                turn_count=turn_count,
                explored_concepts=concepts_str,
                final_understanding=final_understanding
            )
            
            result = self.llm(prompt)
            reflection = result.text.strip() if hasattr(result, "text") else str(result).strip()
            
            # Add metacognitive analysis if requested
            if self.include_metacognitive:
                metacog_analysis = self._generate_metacognitive_analysis(
                    history_str,
                    context
                )
                reflection += metacog_analysis
            
            return AgentOutput(
                text=reflection,
                agent_type=self.agent_type,
                status="finished",
                metadata={
                    "turn_count": turn_count,
                    "explored_concepts": explored_concepts,
                    "final_understanding": final_understanding,
                    "included_metacognitive": self.include_metacognitive,
                }
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