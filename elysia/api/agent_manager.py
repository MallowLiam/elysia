from typing import List, Dict, Any, Type, Optional
from elysia.objects import Tool
import json
import os


class Agent:
    """Represents an agentic agent with tools and configuration."""

    def __init__(
        self,
        name: str,
        description: str,
        tools: List[Type[Tool]],
        system_prompt: str = "",
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = kwargs  # Additional configuration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": [tool.__name__ for tool in self.tools],
            "system_prompt": self.system_prompt,
            "config": self.config,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], tool_registry: Dict[str, Type[Tool]]
    ) -> "Agent":
        tools = [
            tool_registry[tool_name]
            for tool_name in data["tools"]
            if tool_name in tool_registry
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            tools=tools,
            system_prompt=data.get("system_prompt", ""),
            **data.get("config", {}),
        )


class AgentManager:
    """Manages creation, storage, and retrieval of agents."""

    def __init__(self, storage_path: str = "agents.json"):
        self.storage_path = storage_path
        self.agents: Dict[str, Agent] = {}
        self.tool_registry: Dict[str, Type[Tool]] = {}
        self._load_agents()

    def register_tool(self, tool_class: Type[Tool]):
        """Register a tool class for use in agents."""
        self.tool_registry[tool_class.__name__] = tool_class

    def create_agent(
        self,
        name: str,
        description: str,
        tools: List[str],
        system_prompt: str = "",
        **config,
    ) -> Agent:
        """Create a new agent."""
        tool_classes = [
            self.tool_registry[tool_name]
            for tool_name in tools
            if tool_name in self.tool_registry
        ]
        agent = Agent(name, description, tool_classes, system_prompt, **config)
        self.agents[name] = agent
        self._save_agents()
        return agent

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        return [agent.to_dict() for agent in self.agents.values()]

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def update_agent(self, name: str, **updates) -> Agent:
        """Update an existing agent."""
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found.")
        agent = self.agents[name]
        for key, value in updates.items():
            if key == "tools":
                agent.tools = [
                    self.tool_registry[tool_name]
                    for tool_name in value
                    if tool_name in self.tool_registry
                ]
            elif hasattr(agent, key):
                setattr(agent, key, value)
        self._save_agents()
        return agent

    def delete_agent(self, name: str):
        """Delete an agent."""
        if name in self.agents:
            del self.agents[name]
            self._save_agents()

    def _save_agents(self):
        """Save agents to storage."""
        data = {name: agent.to_dict() for name, agent in self.agents.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_agents(self):
        """Load agents from storage."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for name, agent_data in data.items():
                self.agents[name] = Agent.from_dict(agent_data, self.tool_registry)


# Example usage:
# manager = AgentManager()
# manager.register_tool(SafeMath)
# manager.register_tool(EnvironmentSummary)
# agent = manager.create_agent("MathAgent", "An agent for math operations", ["SafeMath"], "You are a math assistant.")
# print(manager.list_agents())
