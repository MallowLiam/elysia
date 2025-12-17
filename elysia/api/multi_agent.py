from typing import Dict, Any, List, Optional, Callable
import asyncio
import json
from datetime import datetime
from elysia.api.agent_manager import AgentManager
from elysia.api.agent_executor import AgentExecutor, ExecutionContext


class Message:
    """Represents a message in the multi-agent communication system."""

    def __init__(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_type = message_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.id = f"{sender}_{recipient}_{int(datetime.now().timestamp())}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class MultiAgentOrchestrator:
    """Orchestrates communication and coordination between multiple agents."""

    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_agents: Dict[str, AgentExecutor] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.conversation_history: List[Message] = []

    async def start_agent(self, agent_name: str) -> bool:
        """Start an agent and add it to the active pool."""
        agent = self.agent_manager.get_agent(agent_name)
        if not agent:
            return False

        executor = AgentExecutor(self.agent_manager)
        self.active_agents[agent_name] = executor

        # Start message processing for this agent
        asyncio.create_task(self._process_agent_messages(agent_name))
        return True

    async def stop_agent(self, agent_name: str) -> bool:
        """Stop an agent and remove it from the active pool."""
        if agent_name in self.active_agents:
            # Clean up any resources
            del self.active_agents[agent_name]
            return True
        return False

    async def send_message(self, message: Message) -> bool:
        """Send a message to an agent."""
        await self.message_queue.put(message)
        self.conversation_history.append(message)
        return True

    async def broadcast_message(
        self, sender: str, content: str, message_type: str = "text"
    ) -> int:
        """Broadcast a message to all active agents."""
        sent_count = 0
        for agent_name in self.active_agents.keys():
            if agent_name != sender:
                message = Message(sender, agent_name, content, message_type)
                await self.send_message(message)
                sent_count += 1
        return sent_count

    async def execute_coordinated_task(
        self,
        task_description: str,
        participating_agents: List[str],
        coordinator_agent: str,
    ) -> Dict[str, Any]:
        """Execute a coordinated task among multiple agents."""

        # Start all participating agents
        for agent_name in participating_agents:
            await self.start_agent(agent_name)

        # Send task to coordinator
        task_message = Message(
            sender="system",
            recipient=coordinator_agent,
            content=f"Coordinate the following task: {task_description}",
            message_type="task",
            metadata={
                "task_type": "coordination",
                "participants": participating_agents,
            },
        )
        await self.send_message(task_message)

        # Wait for responses and coordinate
        responses = []
        timeout = 30  # seconds

        try:
            while len(responses) < len(participating_agents):
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=timeout
                )
                if (
                    message.sender in participating_agents
                    and message.recipient == coordinator_agent
                ):
                    responses.append(message)
        except asyncio.TimeoutError:
            pass

        # Stop agents
        for agent_name in participating_agents:
            await self.stop_agent(agent_name)

        return {
            "task": task_description,
            "coordinator": coordinator_agent,
            "participants": participating_agents,
            "responses": [msg.to_dict() for msg in responses],
            "completed_at": datetime.now().isoformat(),
        }

    async def _process_agent_messages(self, agent_name: str):
        """Process messages for a specific agent."""
        while agent_name in self.active_agents:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                if message.recipient == agent_name:
                    await self._handle_agent_message(agent_name, message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message for {agent_name}: {e}")

    async def _handle_agent_message(self, agent_name: str, message: Message):
        """Handle a message received by an agent."""
        executor = self.active_agents.get(agent_name)
        if not executor:
            return

        # Create execution context for message processing
        context = ExecutionContext(
            agent_name=agent_name,
            task={"type": "message_processing", "message": message.to_dict()},
        )

        # Execute agent with message context
        try:
            result = await executor.execute_agent(
                agent_name=agent_name,
                task={"type": "message_processing", "message": message.to_dict()},
            )

            # Send response back
            response_message = Message(
                sender=agent_name,
                recipient=message.sender,
                content=str(result.get("result", "Message processed")),
                message_type="response",
                metadata={"original_message_id": message.id},
            )
            await self.send_message(response_message)
        except Exception as e:
            error_message = Message(
                sender="system",
                recipient=message.sender,
                content=f"Error processing message: {str(e)}",
                message_type="error",
                metadata={"original_message_id": message.id},
            )
            await self.send_message(error_message)

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a custom message handler."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def get_conversation_history(
        self, agent_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history, optionally filtered by agent."""
        if agent_name:
            return [
                msg.to_dict()
                for msg in self.conversation_history
                if msg.sender == agent_name or msg.recipient == agent_name
            ]
        return [msg.to_dict() for msg in self.conversation_history]

    def get_active_agents(self) -> List[str]:
        """Get list of currently active agents."""
        return list(self.active_agents.keys())

    async def shutdown(self):
        """Shutdown the orchestrator and all active agents."""
        for agent_name in list(self.active_agents.keys()):
            await self.stop_agent(agent_name)
        # Clear message queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AgentCollaboration:
    """Handles collaborative workflows between agents."""

    def __init__(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator
        self.workflows: Dict[str, Dict[str, Any]] = {}

    def define_workflow(
        self, name: str, steps: List[Dict[str, Any]], agents: List[str]
    ):
        """Define a collaborative workflow."""
        self.workflows[name] = {
            "steps": steps,
            "agents": agents,
            "created_at": datetime.now().isoformat(),
        }

    async def execute_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Execute a defined workflow."""
        if workflow_name not in self.workflows:
            return {"error": "Workflow not found"}

        workflow = self.workflows[workflow_name]
        results = []

        for step in workflow["steps"]:
            step_results = await self.orchestrator.execute_coordinated_task(
                task_description=step["description"],
                participating_agents=step["agents"],
                coordinator_agent=step["coordinator"],
            )
            results.append({"step": step["name"], "results": step_results})

        return {
            "workflow": workflow_name,
            "completed_steps": results,
            "completed_at": datetime.now().isoformat(),
        }
