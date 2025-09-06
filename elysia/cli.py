#!/usr/bin/env python3
"""
Elysia Agent Management CLI Tool

A command-line interface for managing agents in the Elysia platform.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elysia.api.agent_manager import AgentManager
from elysia.api.templates import TemplateManager, create_default_templates
from elysia.api.monitoring import MonitoringDashboard
from elysia.api.agent_executor import AgentExecutor


class AgentCLI:
    """Command-line interface for agent management."""

    def __init__(self):
        self.agent_manager = AgentManager()
        self.template_manager = TemplateManager()
        self.agent_executor = AgentExecutor(self.agent_manager)
        self.monitoring = MonitoringDashboard(self.agent_manager, self.agent_executor)

        # Create default templates if they don't exist
        create_default_templates(self.template_manager)

    def confirm_action(self, message: str) -> bool:
        """Get user confirmation for destructive actions."""
        response = input(f"{message} (y/N): ").strip().lower()
        return response in ["y", "yes"]

    def create_agent(self, args):
        """Create a new agent."""
        try:
            agent = self.agent_manager.create_agent(
                name=args.name,
                description=args.description,
                tools=args.tools,
                system_prompt=args.system_prompt,
            )
            print(f"✅ Agent '{args.name}' created successfully!")
            print(json.dumps(agent.to_dict(), indent=2))
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            sys.exit(1)

    def list_agents(self, args):
        """List all agents."""
        agents = self.agent_manager.list_agents()
        if not agents:
            print("No agents found.")
            return

        print(f"📋 Found {len(agents)} agents:")
        for agent in agents:
            status = "🟢 Active" if agent.get("active", False) else "⚪ Inactive"
            print(f"  • {agent['name']} - {agent['description']} ({status})")

    def get_agent(self, args):
        """Get details of a specific agent."""
        agent = self.agent_manager.get_agent(args.name)
        if not agent:
            print(f"❌ Agent '{args.name}' not found.")
            sys.exit(1)

        print(f"📋 Agent '{args.name}':")
        print(json.dumps(agent.to_dict(), indent=2))

    def update_agent(self, args):
        """Update an existing agent."""
        updates = {}
        if args.description:
            updates["description"] = args.description
        if args.tools:
            updates["tools"] = args.tools
        if args.system_prompt:
            updates["system_prompt"] = args.system_prompt

        if not updates:
            print("❌ No updates specified.")
            sys.exit(1)

        try:
            agent = self.agent_manager.update_agent(args.name, **updates)
            if agent:
                print(f"✅ Agent '{args.name}' updated successfully!")
                print(json.dumps(agent.to_dict(), indent=2))
            else:
                print(f"❌ Agent '{args.name}' not found.")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error updating agent: {e}")
            sys.exit(1)

    def delete_agent(self, args):
        """Delete an agent."""
        if not self.confirm_action(
            f"Are you sure you want to delete agent '{args.name}'?"
        ):
            return

        success = self.agent_manager.delete_agent(args.name)
        if success:
            print(f"✅ Agent '{args.name}' deleted successfully!")
        else:
            print(f"❌ Agent '{args.name}' not found.")
            sys.exit(1)

    def execute_agent(self, args):
        """Execute an agent with a task."""
        import asyncio

        async def run_execution():
            try:
                task = {"description": args.task, "parameters": args.parameters or {}}
                result = await self.agent_executor.execute_agent(
                    agent_name=args.name, task=task, timeout_seconds=args.timeout
                )

                # Record for monitoring
                self.monitoring.record_execution(result)

                print(f"🎯 Execution completed for agent '{args.name}':")
                print(json.dumps(result, indent=2))

                if result.get("status") != "completed":
                    sys.exit(1)

            except Exception as e:
                print(f"❌ Error executing agent: {e}")
                sys.exit(1)

        asyncio.run(run_execution())

    def create_from_template(self, args):
        """Create an agent from a template."""
        customizations = {}
        if args.customizations:
            try:
                customizations = json.loads(args.customizations)
            except json.JSONDecodeError:
                print("❌ Invalid JSON for customizations.")
                sys.exit(1)

        result = self.template_manager.create_from_template(
            template_name=args.template,
            customizations=customizations,
            agent_name=args.name,
            agent_manager=self.agent_manager,
        )

        if result:
            print(f"✅ Agent '{args.name}' created from template '{args.template}'!")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Template '{args.template}' not found.")
            sys.exit(1)

    def list_templates(self, args):
        """List available templates."""
        templates = self.template_manager.list_templates()
        if not templates:
            print("No templates found.")
            return

        print(f"📋 Found {len(templates)} templates:")
        for template in templates:
            print(
                f"  • {template['name']} - {template['description']} (v{template['version']})"
            )

    def get_template(self, args):
        """Get details of a specific template."""
        template = self.template_manager.get_template(args.name)
        if not template:
            print(f"❌ Template '{args.name}' not found.")
            sys.exit(1)

        print(f"📋 Template '{args.name}':")
        print(json.dumps(template, indent=2))

    def show_performance(self, args):
        """Show agent performance metrics."""
        if args.agent:
            perf = self.monitoring.get_agent_performance(args.agent)
            if "error" in perf:
                print(f"❌ {perf['error']}")
                sys.exit(1)
            print(f"📊 Performance for agent '{args.agent}':")
        else:
            perf = self.monitoring.get_system_overview()
            print("📊 System Overview:")

        print(json.dumps(perf, indent=2))

    def show_history(self, args):
        """Show execution history."""
        history = self.monitoring.get_execution_history(
            agent_name=args.agent, limit=args.limit, status_filter=args.status
        )

        if not history:
            print("No execution history found.")
            return

        print(f"📋 Execution history ({len(history)} entries):")
        for entry in history:
            status_emoji = {"completed": "✅", "error": "❌", "timeout": "⏰"}.get(
                entry.get("status") or "unknown", "❓"
            )

            print(
                f"  {status_emoji} {entry['agent_name']} - {entry['status']} "
                f"({entry.get('duration', 0):.2f}s)"
            )

    def show_alerts(self, args):
        """Show recent alerts."""
        alerts = self.monitoring.get_alerts(
            severity_filter=args.severity, limit=args.limit
        )

        if not alerts:
            print("No alerts found.")
            return

        print(f"🚨 Recent alerts ({len(alerts)} entries):")
        for alert in alerts:
            severity_emoji = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(
                alert.get("severity") or "unknown", "⚪"
            )

            print(f"  {severity_emoji} {alert['message']}")

    def generate_report(self, args):
        """Generate a performance report."""
        report = self.monitoring.generate_report(time_range_hours=args.hours)
        print("📊 Performance Report:")
        print(json.dumps(report, indent=2))

    def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="Elysia Agent Management CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s create my-agent --description "A helpful assistant"
  %(prog)s execute my-agent --task "Calculate 2+2"
  %(prog)s template-create my-agent --template math_assistant
  %(prog)s performance --agent my-agent
  %(prog)s report --hours 24
            """,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Agent management commands
        create_parser = subparsers.add_parser("create", help="Create a new agent")
        create_parser.add_argument("name", help="Agent name")
        create_parser.add_argument(
            "--description", required=True, help="Agent description"
        )
        create_parser.add_argument("--tools", nargs="*", default=[], help="Agent tools")
        create_parser.add_argument("--system-prompt", help="System prompt")
        create_parser.set_defaults(func=self.create_agent)

        list_parser = subparsers.add_parser("list", help="List all agents")
        list_parser.set_defaults(func=self.list_agents)

        get_parser = subparsers.add_parser("get", help="Get agent details")
        get_parser.add_argument("name", help="Agent name")
        get_parser.set_defaults(func=self.get_agent)

        update_parser = subparsers.add_parser("update", help="Update an agent")
        update_parser.add_argument("name", help="Agent name")
        update_parser.add_argument("--description", help="New description")
        update_parser.add_argument("--tools", nargs="*", help="New tools")
        update_parser.add_argument("--system-prompt", help="New system prompt")
        update_parser.set_defaults(func=self.update_agent)

        delete_parser = subparsers.add_parser("delete", help="Delete an agent")
        delete_parser.add_argument("name", help="Agent name")
        delete_parser.set_defaults(func=self.delete_agent)

        # Execution commands
        execute_parser = subparsers.add_parser("execute", help="Execute an agent")
        execute_parser.add_argument("name", help="Agent name")
        execute_parser.add_argument("--task", required=True, help="Task description")
        execute_parser.add_argument(
            "--parameters", help="Task parameters as JSON string"
        )
        execute_parser.add_argument(
            "--timeout", type=int, default=300, help="Execution timeout in seconds"
        )
        execute_parser.set_defaults(func=self.execute_agent)

        # Template commands
        template_create_parser = subparsers.add_parser(
            "template-create", help="Create agent from template"
        )
        template_create_parser.add_argument("name", help="New agent name")
        template_create_parser.add_argument(
            "--template", required=True, help="Template name"
        )
        template_create_parser.add_argument(
            "--customizations", help="Customizations as JSON string"
        )
        template_create_parser.set_defaults(func=self.create_from_template)

        template_list_parser = subparsers.add_parser(
            "templates", help="List available templates"
        )
        template_list_parser.set_defaults(func=self.list_templates)

        template_get_parser = subparsers.add_parser(
            "template-get", help="Get template details"
        )
        template_get_parser.add_argument("name", help="Template name")
        template_get_parser.set_defaults(func=self.get_template)

        # Monitoring commands
        performance_parser = subparsers.add_parser(
            "performance", help="Show performance metrics"
        )
        performance_parser.add_argument("--agent", help="Specific agent name")
        performance_parser.set_defaults(func=self.show_performance)

        history_parser = subparsers.add_parser("history", help="Show execution history")
        history_parser.add_argument("--agent", help="Filter by agent name")
        history_parser.add_argument(
            "--status",
            choices=["completed", "error", "timeout"],
            help="Filter by status",
        )
        history_parser.add_argument(
            "--limit", type=int, default=20, help="Number of entries to show"
        )
        history_parser.set_defaults(func=self.show_history)

        alerts_parser = subparsers.add_parser("alerts", help="Show recent alerts")
        alerts_parser.add_argument(
            "--severity",
            choices=["error", "warning", "info"],
            help="Filter by severity",
        )
        alerts_parser.add_argument(
            "--limit", type=int, default=20, help="Number of entries to show"
        )
        alerts_parser.set_defaults(func=self.show_alerts)

        report_parser = subparsers.add_parser(
            "report", help="Generate performance report"
        )
        report_parser.add_argument(
            "--hours", type=int, default=24, help="Time range in hours"
        )
        report_parser.set_defaults(func=self.generate_report)

        # Parse arguments
        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            sys.exit(1)

        # Run the selected command
        args.func(args)


def main():
    """Main entry point."""
    try:
        cli = AgentCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
