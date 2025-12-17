from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
from elysia.api.agent_manager import AgentManager


class TemplateManager:
    """Manages agent templates for easy agent creation."""

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_templates()

    def create_template(
        self,
        name: str,
        description: str,
        base_config: Dict[str, Any],
        customization_points: Optional[Dict[str, Any]] = None,
        version: str = "1.0",
    ) -> Dict[str, Any]:
        """Create a new agent template."""

        template = {
            "name": name,
            "description": description,
            "version": version,
            "base_config": base_config,
            "customization_points": customization_points or {},
            "created_at": "2025-09-06T12:00:00Z",
        }

        self.templates[name] = template
        self._save_template(name, template)
        return template

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates."""
        return [
            {
                "name": template["name"],
                "description": template["description"],
                "version": template["version"],
            }
            for template in self.templates.values()
        ]

    def create_from_template(
        self,
        template_name: str,
        customizations: Dict[str, Any],
        agent_name: str,
        agent_manager: AgentManager,
    ) -> Optional[Dict[str, Any]]:
        """Create an agent from a template with customizations."""

        template = self.get_template(template_name)
        if not template:
            return None

        # Start with base config
        agent_config = template["base_config"].copy()

        # Apply customizations
        for key, value in customizations.items():
            if key in template["customization_points"]:
                # Apply customization logic
                point = template["customization_points"][key]
                if point["type"] == "choice" and value in point["options"]:
                    agent_config[key] = value
                elif point["type"] == "text":
                    agent_config[key] = value
                elif point["type"] == "number":
                    agent_config[key] = float(value)

        # Create the agent
        agent = agent_manager.create_agent(
            name=agent_name,
            description=agent_config.get(
                "description", f"Agent created from {template_name} template"
            ),
            tools=agent_config.get("tools", []),
            system_prompt=agent_config.get("system_prompt", ""),
            **{
                k: v
                for k, v in agent_config.items()
                if k not in ["description", "tools", "system_prompt"]
            },
        )

        return {
            "agent": agent.to_dict(),
            "template_used": template_name,
            "customizations_applied": customizations,
        }

    def _save_template(self, name: str, template: Dict[str, Any]):
        """Save a template to disk."""
        template_file = self.templates_dir / f"{name}.json"
        with open(template_file, "w") as f:
            json.dump(template, f, indent=2)

    def _load_templates(self):
        """Load all templates from disk."""
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.json"):
                try:
                    with open(template_file, "r") as f:
                        template = json.load(f)
                    self.templates[template["name"]] = template
                except Exception as e:
                    print(f"Error loading template {template_file}: {e}")

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        if name in self.templates:
            template_file = self.templates_dir / f"{name}.json"
            if template_file.exists():
                template_file.unlink()
            del self.templates[name]
            return True
        return False

    def update_template(
        self, name: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an existing template."""
        if name in self.templates:
            self.templates[name].update(updates)
            self._save_template(name, self.templates[name])
            return self.templates[name]
        return None


# Predefined templates
def create_default_templates(template_manager: TemplateManager):
    """Create some useful default templates."""

    # Math Assistant Template
    template_manager.create_template(
        name="math_assistant",
        description="A specialized agent for mathematical operations",
        base_config={
            "tools": ["SafeMath"],
            "system_prompt": "You are a helpful math assistant.",
            "description": "Mathematical assistant with calculation capabilities",
        },
        customization_points={
            "specialty": {
                "type": "choice",
                "options": ["algebra", "calculus", "statistics", "general"],
                "default": "general",
            },
            "precision": {"type": "number", "default": 2},
        },
    )

    # Data Analyst Template
    template_manager.create_template(
        name="data_analyst",
        description="An agent for data analysis and insights",
        base_config={
            "tools": ["SafeMath", "EnvironmentSummary"],
            "system_prompt": "You are a data analysis expert.",
            "description": "Data analyst with statistical and analytical capabilities",
        },
        customization_points={
            "focus_area": {
                "type": "choice",
                "options": ["business", "scientific", "financial", "general"],
                "default": "general",
            },
            "analysis_depth": {
                "type": "choice",
                "options": ["basic", "detailed", "comprehensive"],
                "default": "detailed",
            },
        },
    )

    # General Assistant Template
    template_manager.create_template(
        name="general_assistant",
        description="A versatile general-purpose assistant",
        base_config={
            "tools": ["SafeMath", "EnvironmentSummary"],
            "system_prompt": "You are a helpful general-purpose assistant.",
            "description": "Versatile assistant for various tasks",
        },
        customization_points={
            "personality": {
                "type": "choice",
                "options": ["professional", "friendly", "concise", "detailed"],
                "default": "professional",
            },
            "expertise_areas": {"type": "text", "default": "general assistance"},
        },
    )
