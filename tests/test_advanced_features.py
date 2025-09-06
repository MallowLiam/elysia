import asyncio
import json
import os
import shutil
from unittest.mock import Mock, patch
from datetime import datetime
from elysia.api.agent_manager import AgentManager
from elysia.api.templates import TemplateManager, create_default_templates
from elysia.api.multi_agent import MultiAgentOrchestrator, Message
from elysia.api.monitoring import MonitoringDashboard
from elysia.api.agent_executor import AgentExecutor, ExecutionContext


class TestTemplateManager:
    """Test cases for TemplateManager."""

    def setup_method(self):
        self.template_manager = TemplateManager(templates_dir="test_templates")

    def teardown_method(self):
        # Clean up test templates
        import shutil

        if os.path.exists("test_templates"):
            shutil.rmtree("test_templates")

    def test_create_template(self):
        """Test creating a new template."""
        template = self.template_manager.create_template(
            name="test_template",
            description="A test template",
            base_config={"tools": ["SafeMath"], "system_prompt": "Test prompt"},
            customization_points={
                "specialty": {"type": "choice", "options": ["math", "text"]}
            },
        )

        assert template["name"] == "test_template"
        assert template["description"] == "A test template"
        assert "SafeMath" in template["base_config"]["tools"]

    def test_get_template(self):
        """Test retrieving a template."""
        self.template_manager.create_template(
            name="test_template",
            description="A test template",
            base_config={"tools": []},
        )

        template = self.template_manager.get_template("test_template")
        assert template is not None
        assert template["name"] == "test_template"

        # Test non-existent template
        assert self.template_manager.get_template("nonexistent") is None

    def test_list_templates(self):
        """Test listing all templates."""
        self.template_manager.create_template(
            name="template1", description="Template 1", base_config={"tools": []}
        )
        self.template_manager.create_template(
            name="template2", description="Template 2", base_config={"tools": []}
        )

        templates = self.template_manager.list_templates()
        assert len(templates) == 2
        assert all("name" in t and "description" in t for t in templates)

    def test_create_from_template(self):
        """Test creating an agent from a template."""
        # Create a mock agent manager
        agent_manager = Mock()
        mock_agent = Mock()
        mock_agent.to_dict.return_value = {
            "name": "test_agent",
            "description": "Test agent",
        }
        agent_manager.create_agent.return_value = mock_agent

        # Create template
        self.template_manager.create_template(
            name="test_template",
            description="A test template",
            base_config={
                "tools": ["SafeMath"],
                "system_prompt": "Test prompt",
                "description": "Base description",
            },
            customization_points={
                "specialty": {
                    "type": "choice",
                    "options": ["math", "text"],
                    "default": "math",
                }
            },
        )

        # Create agent from template
        result = self.template_manager.create_from_template(
            template_name="test_template",
            customizations={"specialty": "text"},
            agent_name="test_agent",
            agent_manager=agent_manager,
        )

        assert result is not None
        agent_manager.create_agent.assert_called_once()


class TestMultiAgentOrchestrator:
    """Test cases for MultiAgentOrchestrator."""

    def setup_method(self):
        self.agent_manager = Mock()
        self.orchestrator = MultiAgentOrchestrator(self.agent_manager)

    def test_send_message(self):
        """Test sending a message."""
        message = Message("sender", "recipient", "test content")
        result = asyncio.run(self.orchestrator.send_message(message))
        assert result is True
        assert len(self.orchestrator.conversation_history) == 1

    def test_broadcast_message(self):
        """Test broadcasting a message."""
        # Mock active agents
        self.orchestrator.active_agents = {"agent1": Mock(), "agent2": Mock()}

        result = asyncio.run(
            self.orchestrator.broadcast_message("sender", "test broadcast")
        )
        assert result == 2  # Should send to 2 agents
        assert len(self.orchestrator.conversation_history) == 2

    def test_get_conversation_history(self):
        """Test retrieving conversation history."""
        message1 = Message("agent1", "agent2", "message 1")
        message2 = Message("agent2", "agent1", "message 2")

        asyncio.run(self.orchestrator.send_message(message1))
        asyncio.run(self.orchestrator.send_message(message2))

        # Get all history
        history = self.orchestrator.get_conversation_history()
        assert len(history) == 2

        # Get filtered history
        history = self.orchestrator.get_conversation_history("agent1")
        assert len(history) == 2  # agent1 sent and received messages


class TestMonitoringDashboard:
    """Test cases for MonitoringDashboard."""

    def setup_method(self):
        self.agent_manager = Mock()
        self.agent_executor = Mock()
        self.monitoring = MonitoringDashboard(self.agent_manager, self.agent_executor)

    def test_record_execution(self):
        """Test recording an execution."""
        execution_result = {
            "execution_id": "test_123",
            "agent_name": "test_agent",
            "status": "completed",
            "duration": 1.5,
            "steps": 5,
        }

        self.monitoring.record_execution(execution_result)

        assert "test_agent" in self.monitoring.performance_data
        perf = self.monitoring.performance_data["test_agent"]
        assert perf["total_executions"] == 1
        assert perf["successful_executions"] == 1
        assert perf["average_duration"] == 1.5

    def test_get_agent_performance(self):
        """Test getting agent performance metrics."""
        # Record some executions
        executions = [
            {
                "execution_id": "1",
                "agent_name": "test_agent",
                "status": "completed",
                "duration": 1.0,
            },
            {
                "execution_id": "2",
                "agent_name": "test_agent",
                "status": "error",
                "duration": 0.5,
            },
            {
                "execution_id": "3",
                "agent_name": "test_agent",
                "status": "completed",
                "duration": 2.0,
            },
        ]

        for execution in executions:
            self.monitoring.record_execution(execution)

        perf = self.monitoring.get_agent_performance("test_agent")

        assert perf["total_executions"] == 3
        assert perf["successful_executions"] == 2
        assert perf["failed_executions"] == 1
        assert perf["success_rate"] == "66.7%"
        assert perf["average_duration"] == "1.17s"

    def test_get_system_overview(self):
        """Test getting system overview."""
        # Mock agent manager
        self.agent_manager.list_agents.return_value = [
            {"name": "agent1"},
            {"name": "agent2"},
        ]
        self.agent_executor.active_executions = {"exec1": Mock()}

        overview = self.monitoring.get_system_overview()

        assert overview["total_agents"] == 2
        assert overview["active_agents"] == 1
        assert "system_health" in overview

    def test_generate_report(self):
        """Test generating a performance report."""
        # Record some executions
        executions = [
            {
                "execution_id": "1",
                "agent_name": "agent1",
                "status": "completed",
                "duration": 1.0,
                "timestamp": datetime.now().isoformat(),
            },
            {
                "execution_id": "2",
                "agent_name": "agent2",
                "status": "completed",
                "duration": 2.0,
                "timestamp": datetime.now().isoformat(),
            },
        ]

        for execution in executions:
            self.monitoring.record_execution(execution)

        report = self.monitoring.generate_report(time_range_hours=24)

        assert report["total_executions"] == 2
        assert report["successful_executions"] == 2
        assert "success_rate" in report
        assert "top_performing_agents" in report


class TestExecutionContext:
    """Test cases for ExecutionContext."""

    def test_execution_context_creation(self):
        """Test creating an execution context."""
        task = {"description": "Test task", "parameters": {"key": "value"}}
        context = ExecutionContext("test_agent", task)

        assert context.agent_name == "test_agent"
        assert context.task == task
        assert context.start_time is not None
        assert context.end_time is None
        assert context.steps == []
        assert context.execution_id.startswith("test_agent_")

    def test_add_step(self):
        """Test adding execution steps."""
        context = ExecutionContext("test_agent", {"description": "Test"})

        context.add_step("processing", {"data": "test"})
        context.add_step("completion", {"result": "success"})

        assert len(context.steps) == 2
        assert context.steps[0]["type"] == "processing"
        assert context.steps[1]["type"] == "completion"

    def test_complete_execution(self):
        """Test completing an execution."""
        context = ExecutionContext("test_agent", {"description": "Test"})

        context.complete(True, "success result")

        assert context.end_time is not None
        assert len(context.steps) == 1
        assert context.steps[0]["type"] == "completion"
        assert context.steps[0]["data"]["success"] is True
        assert context.steps[0]["data"]["result"] == "success result"


class TestAgentExecutorIntegration:
    """Integration tests for AgentExecutor."""

    def setup_method(self):
        self.agent_manager = AgentManager()
        self.executor = AgentExecutor(self.agent_manager)

    @patch("elysia.api.agent_executor.AgentExecutor._execute_agent_internal")
    def test_execute_agent_success(self, mock_execute):
        """Test successful agent execution."""
        # Mock the internal execution
        mock_execute.return_value = {"result": "success", "steps": 3}

        # Create a mock agent
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        self.agent_manager.agents = {"test_agent": mock_agent}
        self.agent_manager.get_agent = Mock(return_value=mock_agent)

        async def run_test():
            result = await self.executor.execute_agent(
                agent_name="test_agent", task={"description": "Test task"}
            )

            assert result["status"] == "completed"
            assert result["result"] == {"result": "success", "steps": 3}
            assert "execution_id" in result
            assert "duration" in result

        asyncio.run(run_test())

    def test_execute_agent_not_found(self):
        """Test execution with non-existent agent."""

        async def run_test():
            try:
                await self.executor.execute_agent(
                    agent_name="nonexistent", task={"description": "Test task"}
                )
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Agent 'nonexistent' not found" in str(e)

        asyncio.run(run_test())


# Integration test for the complete system
class TestElysiaIntegration:
    """Integration tests for the complete Elysia system."""

    def setup_method(self):
        self.agent_manager = AgentManager()
        self.template_manager = TemplateManager()
        self.executor = AgentExecutor(self.agent_manager)
        self.monitoring = MonitoringDashboard(self.agent_manager, self.executor)

        # Create default templates
        create_default_templates(self.template_manager)

    def test_full_agent_lifecycle(self):
        """Test the complete agent lifecycle from template to execution."""
        # Create agent from template
        result = self.template_manager.create_from_template(
            template_name="math_assistant",
            customizations={"specialty": "algebra"},
            agent_name="math_agent",
            agent_manager=self.agent_manager,
        )

        assert result is not None
        assert "math_agent" in [
            agent["name"] for agent in self.agent_manager.list_agents()
        ]

        # Execute the agent
        async def run_execution():
            execution_result = await self.executor.execute_agent(
                agent_name="math_agent", task={"description": "Calculate 2 + 2"}
            )

            # Record for monitoring
            self.monitoring.record_execution(execution_result, agent_name="math_agent")

            assert execution_result["status"] in ["completed", "error"]
            assert "execution_id" in execution_result

        asyncio.run(run_execution())

        # Check monitoring data
        perf = self.monitoring.get_agent_performance("math_agent")
        assert perf["total_executions"] == 1

    def test_template_system(self):
        """Test the template system end-to-end."""
        # List available templates
        templates = self.template_manager.list_templates()
        assert len(templates) >= 3  # Should have default templates

        # Get a specific template
        template = self.template_manager.get_template("math_assistant")
        assert template is not None
        assert "customization_points" in template

        # Create agent from template
        result = self.template_manager.create_from_template(
            template_name="math_assistant",
            customizations={},
            agent_name="templated_agent",
            agent_manager=self.agent_manager,
        )

        assert result is not None
        assert result["template_used"] == "math_assistant"


# Simple test runner
def run_tests():
    """Run all tests manually."""
    print("Running Elysia Advanced Features Tests...")

    # Test TemplateManager
    print("\n1. Testing TemplateManager...")
    template_test = TestTemplateManager()
    try:
        template_test.setup_method()
        template_test.test_create_template()
        template_test.test_get_template()
        template_test.test_list_templates()
        template_test.test_create_from_template()
        template_test.teardown_method()
        print("   ✅ TemplateManager tests passed")
    except Exception as e:
        print(f"   ❌ TemplateManager test failed: {e}")

    # Test MultiAgentOrchestrator
    print("\n2. Testing MultiAgentOrchestrator...")
    orchestrator_test = TestMultiAgentOrchestrator()
    try:
        orchestrator_test.setup_method()
        orchestrator_test.test_send_message()
        orchestrator_test.test_broadcast_message()
        orchestrator_test.test_get_conversation_history()
        print("   ✅ MultiAgentOrchestrator tests passed")
    except Exception as e:
        print(f"   ❌ MultiAgentOrchestrator test failed: {e}")

    # Test MonitoringDashboard
    print("\n3. Testing MonitoringDashboard...")
    monitoring_test = TestMonitoringDashboard()
    try:
        monitoring_test.setup_method()
        monitoring_test.test_record_execution()
        monitoring_test.test_get_agent_performance()
        monitoring_test.test_get_system_overview()
        monitoring_test.test_generate_report()
        print("   ✅ MonitoringDashboard tests passed")
    except Exception as e:
        print(f"   ❌ MonitoringDashboard test failed: {e}")

    # Test ExecutionContext
    print("\n4. Testing ExecutionContext...")
    context_test = TestExecutionContext()
    try:
        context_test.test_execution_context_creation()
        context_test.test_add_step()
        context_test.test_complete_execution()
        print("   ✅ ExecutionContext tests passed")
    except Exception as e:
        print(f"   ❌ ExecutionContext test failed: {e}")

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    run_tests()
