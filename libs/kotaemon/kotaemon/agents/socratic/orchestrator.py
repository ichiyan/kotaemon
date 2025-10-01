from typing import Optional, Dict, List, Any
from kotaemon.agents.base import BaseAgent, AgentOutput, AgentType
from kotaemon.llms import BaseLLM, ChatLLM


class SocraticOrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that manages the Socratic dialogue flow, tracks state,
    and decides when to transition between agents.
    """

    name: str = "SocraticOrchestrator"
    agent_type: AgentType = AgentType.socratic_orchestrator
    description: str = "Manages Socratic dialogue flow and agent transitions"

    class Config:
        allow_extra = True

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_turns: int = 8,
        stuck_threshold: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm or ChatLLM.default()
        self.max_turns = max_turns
        self.stuck_threshold = stuck_threshold
        
        # Dialogue state tracking
        self.reset_state()

    def reset_state(self):
        """Reset orchestrator state for new dialogue"""
        self.turn_number = 0
        self.explored_concepts = []
        self.confusion_count = 0
        self.partial_count = 0
        self.dialogue_history = []
        self.ground_truth_context = ""

    def update_state(self, student_reply: str, agent_response: str, metadata: Dict[str, Any]):
        """Update orchestrator state after each exchange"""
        self.turn_number += 1
        
        # Add to history
        self.dialogue_history.append({
            "role": "student",
            "content": student_reply
        })
        self.dialogue_history.append({
            "role": "tutor",
            "content": agent_response
        })
        
        # Update tracked concepts
        if "explored_concepts" in metadata:
            self.explored_concepts = metadata["explored_concepts"]
        
        # Track confusion patterns
        understanding = metadata.get("understanding_level", "partial")
        if understanding == "confused":
            self.confusion_count += 1
        else:
            self.confusion_count = 0  # Reset if they recover
            
        if understanding == "partial":
            self.partial_count += 1
        else:
            self.partial_count = 0

    def should_give_hint(self, evaluation_decision: str, understanding_level: str) -> bool:
        """Decide if student needs a hint"""
        return (
            evaluation_decision == "hint" or
            self.confusion_count >= self.stuck_threshold or
            (understanding_level == "confused" and self.turn_number >= self.stuck_threshold)
        )

    def should_reflect(self, evaluation_decision: str, understanding_level: str) -> bool:
        """Decide if dialogue should end with reflection"""
        return (
            evaluation_decision == "reflect" or
            self.turn_number >= self.max_turns or
            (understanding_level == "strong" and len(self.explored_concepts) >= 2)
        )

    def should_continue_socratic(self, evaluation_decision: str) -> bool:
        """Decide if Socratic dialogue should continue"""
        return (
            evaluation_decision == "continue_socratic" and
            self.turn_number < self.max_turns and
            self.confusion_count < self.stuck_threshold
        )

    def get_next_action(self, evaluator_output: AgentOutput, socratic_metadata: Dict[str, Any]) -> str:
        """
        Determine next action based on evaluator decision and current state.
        
        Returns: "continue_socratic" | "hint" | "reflect"
        """
        evaluation_decision = evaluator_output.text.lower().strip()
        understanding_level = socratic_metadata.get("understanding_level", "partial")
        
        # Priority order: reflect > hint > continue
        if self.should_reflect(evaluation_decision, understanding_level):
            return "reflect"
        elif self.should_give_hint(evaluation_decision, understanding_level):
            return "hint"
        elif self.should_continue_socratic(evaluation_decision):
            return "continue_socratic"
        else:
            # Fallback: if uncertain, continue or reflect based on turns
            return "reflect" if self.turn_number >= self.max_turns else "continue_socratic"

    def prepare_agent_context(self, student_reply: str, agent_type: str) -> Dict[str, Any]:
        """Prepare context for specific agent invocation"""
        base_context = {
            "context": self.ground_truth_context,
            "student_reply": student_reply,
            "history": self.dialogue_history.copy(),
            "turn_number": self.turn_number,
            "explored_concepts": self.explored_concepts.copy(),
        }
        
        if agent_type == "socratic":
            # Additional context for Socratic agent
            base_context.update({
                "max_turns": self.max_turns,
                "stuck_threshold": self.stuck_threshold,
            })
        
        return base_context

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for logging/debugging"""
        return {
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "explored_concepts": self.explored_concepts,
            "confusion_count": self.confusion_count,
            "partial_count": self.partial_count,
            "dialogue_length": len(self.dialogue_history),
        }

    def run(self, instruction: dict) -> AgentOutput:
        """
        Main orchestration logic.
        
        Args:
            instruction: dict with:
                - action: "init" | "next_action"
                - ground_truth_context: context from ReAct agent (for init)
                - student_reply: current student message
                - evaluator_output: output from evaluator agent
                - socratic_metadata: metadata from socratic agent
        
        Returns:
            AgentOutput with next_action decision
        """
        try:
            action = instruction.get("action", "next_action")
            
            if action == "init":
                # Initialize new dialogue
                self.reset_state()
                self.ground_truth_context = instruction.get("ground_truth_context", "")
                return AgentOutput(
                    text="continue_socratic",
                    agent_type=self.agent_type,
                    status="finished",
                    metadata={"initialized": True}
                )
            
            elif action == "next_action":
                # Determine next step in dialogue
                evaluator_output = instruction.get("evaluator_output")
                socratic_metadata = instruction.get("socratic_metadata", {})
                student_reply = instruction.get("student_reply", "")
                agent_response = instruction.get("agent_response", "")
                
                # Update state
                self.update_state(student_reply, agent_response, socratic_metadata)
                
                # Decide next action
                next_action = self.get_next_action(evaluator_output, socratic_metadata)
                
                return AgentOutput(
                    text=next_action,
                    agent_type=self.agent_type,
                    status="finished",
                    metadata={
                        "state_summary": self.get_state_summary(),
                        "next_action": next_action,
                        "turn_number": self.turn_number,
                    }
                )
            
            else:
                return AgentOutput(
                    text="",
                    agent_type=self.agent_type,
                    status="failed",
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return AgentOutput(
                text="",
                agent_type=self.agent_type,
                status="failed",
                error=str(e),
            )