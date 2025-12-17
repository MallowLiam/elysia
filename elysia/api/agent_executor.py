from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
import time
import logging
from datetime import datetime
from elysia.api.agent_manager import AgentManager
from elysia.objects import Tool, Result, Status, Error

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Context for agent execution with state management."""

    def __init__(self, agent_name: str, task: Dict[str, Any]):
        self.agent_name = agent_name
        self.task = task
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.steps: List[Dict[str, Any]] = []
        self.environment: Dict[str, Any] = {}
        self.hidden_environment: Dict[str, Any] = {}
        self.execution_id = f"{agent_name}_{int(time.time())}"

    def add_step(self, step_type: str, data: Dict[str, Any]):
        """Add an execution step to the context."""
        step = {"timestamp": datetime.now(), "type": step_type, "data": data}
        self.steps.append(step)

    def complete(self, success: bool, result: Any = None):
        """Mark execution as complete."""
        self.end_time = datetime.now()
        self.add_step(
            "completion",
            {
                "success": success,
                "result": result,
                "duration": (self.end_time - self.start_time).total_seconds(),
            },
        )


class AgentExecutor:
    """Executes agents with proper state management and monitoring."""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionContext] = []

    async def execute_agent(
        self,
        agent_name: str,
        task: Dict[str, Any],
        timeout_seconds: int = 300,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Execute an agent with the given task."""

        # Get the agent
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        # Create execution context
        context = ExecutionContext(agent_name, task)
        self.active_executions[context.execution_id] = context

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_agent_internal(agent, task, context, max_steps),
                timeout=timeout_seconds,
            )

            context.complete(True, result)
            return {
                "execution_id": context.execution_id,
                "status": "completed",
                "result": result,
                "steps": len(context.steps),
                "duration": (
                    (context.end_time - context.start_time).total_seconds()
                    if context.end_time
                    else 0
                ),
            }

        except asyncio.TimeoutError:
            context.complete(False, "timeout")
            return {
                "execution_id": context.execution_id,
                "status": "timeout",
                "error": f"Execution exceeded {timeout_seconds} seconds",
                "steps": len(context.steps),
            }

        except Exception as e:
            context.complete(False, str(e))
            return {
                "execution_id": context.execution_id,
                "status": "error",
                "error": str(e),
                "steps": len(context.steps),
            }

        finally:
            # Clean up active executions
            if context.execution_id in self.active_executions:
                del self.active_executions[context.execution_id]
            self.execution_history.append(context)

    async def _execute_agent_internal(
        self, agent, task: Dict[str, Any], context: ExecutionContext, max_steps: int
    ) -> Dict[str, Any]:
        """Internal agent execution logic."""

        context.add_step("start", {"task": task})

        # For now, implement basic tool execution
        # This can be extended to support decision trees and complex workflows

        objective = task.get("objective", "")
        inputs = task.get("inputs", {})

        # Find appropriate tool and execute
        result = await self._execute_with_tools(agent, inputs, context)

        context.add_step("result", {"result": result})
        return result

    async def _execute_with_tools(
        self, agent, inputs: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute using agent's tools."""

        # Mock tree data structure (would come from actual decision tree)
        class MockTreeData:
            def __init__(self, context):
                self.environment = type(
                    "obj",
                    (object,),
                    {
                        "environment": context.environment,
                        "hidden_environment": context.hidden_environment,
                    },
                )()

        tree_data = MockTreeData(context)

        # Try to execute with available tools
        for tool_class in agent.tools:
            try:
                tool_instance = tool_class()
                context.add_step(
                    "tool_execution", {"tool": tool_class.__name__, "inputs": inputs}
                )

                # Execute tool
                results = []
                async for result in tool_instance(
                    tree_data,
                    inputs,
                    None,  # base_lm
                    None,  # complex_lm
                    None,  # client_manager
                ):
                    results.append(result)

                # Process results
                for result in results:
                    if isinstance(result, Result):
                        context.add_step(
                            "tool_result",
                            {
                                "tool": tool_class.__name__,
                                "result": result.objects[0] if result.objects else None,
                            },
                        )
                        return (
                            result.objects[0]
                            if result.objects
                            else {"message": "No result"}
                        )

            except Exception as e:
                context.add_step(
                    "tool_error", {"tool": tool_class.__name__, "error": str(e)}
                )
                continue

        return {"error": "No suitable tool found or all tools failed"}

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution."""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                "execution_id": execution_id,
                "status": "running",
                "agent": context.agent_name,
                "steps": len(context.steps),
                "duration": (datetime.now() - context.start_time).total_seconds(),
            }
        return None

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        recent = self.execution_history[-limit:]
        return [
            {
                "execution_id": ctx.execution_id,
                "agent": ctx.agent_name,
                "status": "completed" if ctx.end_time else "running",
                "steps": len(ctx.steps),
                "duration": (
                    (ctx.end_time - ctx.start_time).total_seconds()
                    if ctx.end_time
                    else None
                ),
            }
            for ctx in recent
        ]

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            context.complete(False, "cancelled")
            del self.active_executions[execution_id]
            return True
        return False
