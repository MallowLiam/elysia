from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from elysia.api.agent_manager import AgentManager
from elysia.api.agent_executor import AgentExecutor


class MonitoringDashboard:
    """Dashboard for monitoring agent performance and analytics."""

    def __init__(self, agent_manager: AgentManager, agent_executor: AgentExecutor):
        self.agent_manager = agent_manager
        self.agent_executor = agent_executor
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, Dict[str, Any]] = {}

    def record_execution(
        self, execution_result: Dict[str, Any], agent_name: Optional[str] = None
    ):
        """Record an agent execution for monitoring."""
        agent_name = agent_name or execution_result.get("agent_name", "unknown")
        if not agent_name:
            agent_name = "unknown"
        timestamp = datetime.now().isoformat()

        metric = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            "execution_id": execution_result.get("execution_id"),
            "status": execution_result.get("status"),
            "duration": execution_result.get("duration", 0),
            "steps": execution_result.get("steps", 0),
            "error": execution_result.get("error"),
        }

        self.metrics[agent_name].append(metric)

        # Update performance data
        if agent_name not in self.performance_data:
            self.performance_data[agent_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_duration": 0,
                "average_duration": 0,
                "last_execution": None,
            }

        perf = self.performance_data[agent_name]
        perf["total_executions"] += 1
        perf["last_execution"] = timestamp

        if execution_result.get("status") == "completed":
            perf["successful_executions"] += 1
        else:
            perf["failed_executions"] += 1

        perf["total_duration"] += execution_result.get("duration", 0)
        perf["average_duration"] = perf["total_duration"] / perf["total_executions"]

        # Check for alerts
        self._check_alerts(agent_name, execution_result)

    def _check_alerts(self, agent_name: str, execution_result: Dict[str, Any]):
        """Check for performance alerts."""
        duration = execution_result.get("duration", 0)
        status = execution_result.get("status")

        # Alert on long executions
        if duration > 60:  # More than 1 minute
            self.alerts.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "performance",
                    "severity": "warning",
                    "message": f"Agent {agent_name} took {duration:.2f}s to execute",
                    "agent": agent_name,
                    "execution_id": execution_result.get("execution_id"),
                }
            )

        # Alert on failures
        if status in ["error", "timeout"]:
            self.alerts.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "error",
                    "severity": "error",
                    "message": f"Agent {agent_name} execution failed: {execution_result.get('error', 'Unknown error')}",
                    "agent": agent_name,
                    "execution_id": execution_result.get("execution_id"),
                }
            )

    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent."""
        if agent_name not in self.performance_data:
            return {"error": "Agent not found"}

        perf = self.performance_data[agent_name]
        success_rate = (
            (perf["successful_executions"] / perf["total_executions"]) * 100
            if perf["total_executions"] > 0
            else 0
        )

        return {
            "agent_name": agent_name,
            "total_executions": perf["total_executions"],
            "successful_executions": perf["successful_executions"],
            "failed_executions": perf["failed_executions"],
            "success_rate": f"{success_rate:.1f}%",
            "average_duration": f"{perf['average_duration']:.2f}s",
            "last_execution": perf["last_execution"],
        }

    def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system performance overview."""
        total_agents = len(self.agent_manager.list_agents())
        active_agents = len(self.agent_executor.active_executions)
        total_executions = sum(len(metrics) for metrics in self.metrics.values())
        recent_alerts = [alert for alert in self.alerts[-10:]]  # Last 10 alerts

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "total_executions": total_executions,
            "recent_alerts": recent_alerts,
            "system_health": self._calculate_system_health(),
        }

    def _calculate_system_health(self) -> str:
        """Calculate overall system health status."""
        if not self.performance_data:
            return "unknown"

        total_executions = sum(
            perf["total_executions"] for perf in self.performance_data.values()
        )
        successful_executions = sum(
            perf["successful_executions"] for perf in self.performance_data.values()
        )

        if total_executions == 0:
            return "unknown"

        success_rate = (successful_executions / total_executions) * 100

        if success_rate >= 95:
            return "excellent"
        elif success_rate >= 85:
            return "good"
        elif success_rate >= 70:
            return "fair"
        else:
            return "poor"

    def get_execution_history(
        self,
        agent_name: Optional[str] = None,
        limit: int = 50,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get execution history with optional filtering."""
        history = []

        if agent_name:
            metrics = self.metrics.get(agent_name, [])
        else:
            metrics = []
            for agent_metrics in self.metrics.values():
                metrics.extend(agent_metrics)

        # Sort by timestamp (most recent first)
        metrics.sort(key=lambda x: x["timestamp"], reverse=True)

        for metric in metrics[:limit]:
            if status_filter and metric.get("status") != status_filter:
                continue
            history.append(metric)

        return history

    def get_alerts(
        self, severity_filter: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent alerts with optional severity filtering."""
        alerts = self.alerts[-limit:]

        if severity_filter:
            alerts = [
                alert for alert in alerts if alert.get("severity") == severity_filter
            ]

        return alerts

    def generate_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        # Filter metrics within time range
        recent_metrics = []
        for agent_metrics in self.metrics.values():
            for metric in agent_metrics:
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if metric_time >= cutoff_time:
                    recent_metrics.append(metric)

        # Calculate statistics
        total_executions = len(recent_metrics)
        successful = len([m for m in recent_metrics if m.get("status") == "completed"])
        failed = len(
            [m for m in recent_metrics if m.get("status") in ["error", "timeout"]]
        )

        avg_duration = (
            sum(m.get("duration", 0) for m in recent_metrics) / total_executions
            if total_executions > 0
            else 0
        )

        return {
            "time_range": f"{time_range_hours} hours",
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": (
                f"{(successful / total_executions * 100):.1f}%"
                if total_executions > 0
                else "0%"
            ),
            "average_duration": f"{avg_duration:.2f}s",
            "top_performing_agents": self._get_top_agents(recent_metrics),
            "generated_at": datetime.now().isoformat(),
        }

    def _get_top_agents(
        self, metrics: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top performing agents based on success rate."""
        agent_stats = defaultdict(lambda: {"total": 0, "successful": 0})

        for metric in metrics:
            agent = metric.get("agent_name", "unknown")
            agent_stats[agent]["total"] += 1
            if metric.get("status") == "completed":
                agent_stats[agent]["successful"] += 1

        # Calculate success rates and sort
        agent_performance = []
        for agent, stats in agent_stats.items():
            success_rate = (
                (stats["successful"] / stats["total"]) * 100
                if stats["total"] > 0
                else 0
            )
            agent_performance.append(
                {
                    "agent": agent,
                    "success_rate": f"{success_rate:.1f}%",
                    "total_executions": stats["total"],
                }
            )

        agent_performance.sort(
            key=lambda x: float(x["success_rate"].rstrip("%")), reverse=True
        )
        return agent_performance[:top_n]

    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old monitoring data to prevent memory issues."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Clear old metrics
        for agent_name in self.metrics:
            self.metrics[agent_name] = [
                metric
                for metric in self.metrics[agent_name]
                if datetime.fromisoformat(metric["timestamp"]) >= cutoff_date
            ]

        # Clear old alerts
        self.alerts = [
            alert
            for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) >= cutoff_date
        ]
