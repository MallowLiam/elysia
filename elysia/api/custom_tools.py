from elysia.objects import Tool, Response

# Optional imports (heavy/extra deps). Keep non-fatal if unavailable so that
# lightweight tools in this module remain usable in no-deps environments.
try:  # visualisation tools may depend on plotting libs
    from elysia.tools.visualisation.linear_regression import BasicLinearRegression  # type: ignore
except Exception:  # pragma: no cover - optional
    BasicLinearRegression = None  # type: ignore

try:  # retrieval stack may require external services
    from elysia.tools.retrieval.query import Query  # type: ignore
    from elysia.tools.retrieval.aggregate import Aggregate  # type: ignore
except Exception:  # pragma: no cover - optional
    Query = None  # type: ignore
    Aggregate = None  # type: ignore

try:  # text tools pull in LLM/dspy deps
    from elysia.tools.text.text import CitedSummarizer, FakeTextResponse  # type: ignore
except Exception:  # pragma: no cover - optional
    CitedSummarizer = None  # type: ignore
    FakeTextResponse = None  # type: ignore


# Or you can define the tool inline here
class TellAJoke(Tool):
    """
    Example tool for testing/demonstration purposes.
    Simply returns a joke as a text response that was an input to the tool.
    """

    def __init__(self, **kwargs):

        # Init requires initialisation of the super class (Tool)
        super().__init__(
            name="tell_a_joke",
            description="Displays a joke to the user.",
            inputs={
                "joke": {
                    "type": str,
                    "description": "A joke to tell.",
                    "required": True,
                }
            },
            end=True,
        )

    # Call must be a async generator function that yields objects to the decision tree
    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):

        # This example tool only returns the input to the tool, so is not very useful
        yield Response(inputs["joke"])

        # You can include more complex logic here via a custom function


# ---------------------------------------------------------------------------
# Advanced Example Tools
# These demonstrate: environment access, hidden environment usage, error
# handling, dynamic availability, flexible input assignment and multi-yield
# status/result patterns.
# ---------------------------------------------------------------------------

from elysia.objects import Result, Status, Error
from typing import Any, Union


class SafeMath(Tool):
    """Perform a mathematical aggregation on a list of numbers.

    Demonstrates:
    - Flexible input assignment (list of numbers passed as comma / space separated string)
    - Error handling via yielding `Error`
    - Returning structured data with a `Result`
    - Allowing the tool to optionally terminate the tree (`end=True` when a
      decisive computation answers the user’s query)
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="safe_math",
            description=(
                "Safely perform a mathematical operation over a list of numbers. "
                "Supported operations: sum, product, min, max, mean. Input numbers can be a list or a comma/space separated string."
            ),
            inputs={
                "operation": {
                    "type": str,
                    "description": "The aggregation to perform: sum | product | min | max | mean",
                    "required": True,
                },
                "numbers": {
                    "type": Union[list, str],
                    "description": "Numbers as list OR comma / space separated string",
                    "required": True,
                },
            },
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        raw_numbers = inputs["numbers"]
        numbers = []
        if isinstance(raw_numbers, str):
            # split by comma or whitespace
            parts = (
                [p.strip() for chunk in raw_numbers.split(",") for p in chunk.split()]
                if "," in raw_numbers
                else raw_numbers.split()
            )
            for p in parts:
                try:
                    numbers.append(float(p))
                except ValueError:
                    yield Error(f"Invalid number: '{p}' is not a valid float.")
                    return
        elif isinstance(raw_numbers, list):
            for p in raw_numbers:
                try:
                    numbers.append(float(p))
                except Exception:
                    yield Error(f"Invalid list element: '{p}' is not a valid float.")
                    return
        else:
            yield Error("Unsupported numbers input format.")
            return

        if not numbers:
            yield Error("No numbers provided.")
            return

        operation = inputs["operation"].lower().strip()
        from math import prod

        try:
            match operation:
                case "sum":
                    value = sum(numbers)
                case "product":
                    value = prod(numbers)
                case "min":
                    value = min(numbers)
                case "max":
                    value = max(numbers)
                case "mean":
                    value = sum(numbers) / len(numbers)
                case _:
                    yield Error(
                        f"Unsupported operation '{operation}'. Use one of: sum, product, min, max, mean."
                    )
                    return
        except Exception as e:
            yield Error(f"Failed to compute operation: {e}")
            return

        yield Status(f"Computed {operation} over {len(numbers)} numbers successfully.")

        yield Result(
            objects=[{"operation": operation, "value": value, "count": len(numbers)}],
            metadata={"numbers": numbers},
            name="math_result",
            llm_message=(
                "Performed {operation} over {count} numbers. Result value: {value}."
            ),
        )


class EnvironmentSummary(Tool):
    """Summarise the current environment contents.

    Demonstrates reading from `tree_data.environment` and producing a structured
    summary. Useful for debugging or deciding next steps.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="environment_summary",
            description="Review and summarise what data is currently stored in the decision tree environment.",
            inputs={},
            end=False,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        env = tree_data.environment.environment
        summary: list[dict[str, Any]] = []
        for tool_name, results in env.items():
            if tool_name == "SelfInfo":
                continue
            for result_name, result_list in results.items():
                summary.append(
                    {
                        "tool": tool_name,
                        "result_name": result_name,
                        "num_batches": len(result_list),
                        "total_objects": sum(
                            len(r.get("objects", [])) for r in result_list
                        ),
                    }
                )

        yield Status("Calculated environment summary.")
        yield Result(
            objects=summary if summary else [{"message": "Environment empty"}],
            metadata={"total_entries": len(summary)},
            name="environment_summary",
            llm_message="Environment contains {total_entries} result group(s).",
        )


class HiddenStoreWriter(Tool):
    """Store arbitrary key/value data into the hidden environment.

    Demonstrates writing to `tree_data.environment.hidden_environment` so that
    other tools can react to internal state not exposed directly to the model.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="hidden_store_writer",
            description="Persist a key/value pair into the hidden environment store.",
            inputs={
                "key": {"type": str, "description": "Key to store", "required": True},
                "value": {
                    "type": str,
                    "description": "Value to store",
                    "required": True,
                },
            },
            end=False,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        key = inputs["key"]
        value = inputs["value"]
        tree_data.environment.hidden_environment[key] = value
        yield Status(f"Stored hidden key '{key}'.")
        yield Result(
            objects=[{"stored_key": key, "stored_value": value}],
            metadata={"hidden_count": len(tree_data.environment.hidden_environment)},
            name="hidden_store_write",
            llm_message="Stored hidden key; total hidden entries now {hidden_count}.",
        )


class HiddenStoreConditionalTool(Tool):
    """Only available if a specific hidden key/value pair exists.

    Demonstrates dynamic availability by overriding `is_tool_available`.
    If available, it reports the stored value and can end the conversation.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="hidden_store_reader",
            description=(
                "Conditionally available tool that reads a hidden environment key "
                "and returns its value if present. Requires that 'unlock' key exists."
            ),
            inputs={},
            end=True,
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return "unlock" in tree_data.environment.hidden_environment

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        value = tree_data.environment.hidden_environment.get("unlock")
        if value is None:
            # Should not normally be callable if not available, but guard anyway
            yield Error("Hidden key 'unlock' not present.")
            return
        yield Status("Retrieved hidden unlock value.")
        yield Result(
            objects=[{"unlock": value}],
            metadata={
                "available_keys": list(tree_data.environment.hidden_environment.keys())
            },
            name="hidden_store_read",
            llm_message="Unlock value retrieved; keys available: {available_keys}.",
        )


# Registering these tools is automatic: they are discovered by the API via
# `find_tool_classes()` scanning this module for subclasses of `Tool`.


class QdrantSearch(Tool):
    """Search a Qdrant collection with a provided query vector.

    Inputs:
    - collection: Name of the Qdrant collection
    - query_vector: List of floats OR a comma/space separated string
    - limit: Max number of results to return (default 10)

    This tool is only available when `VECTOR_DB_TYPE=qdrant` is set.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="qdrant_search",
            description=(
                "Search a Qdrant collection using a numeric query vector. "
                "Inputs: collection (str), query_vector (list[str|float] or comma/space separated str), limit (int)."
            ),
            inputs={
                "collection": {
                    "type": str,
                    "description": "Qdrant collection name",
                    "required": True,
                },
                "query_vector": {
                    "type": list,
                    "description": "Query vector as list OR comma / space separated string",
                    "required": True,
                },
                "limit": {
                    "type": int,
                    "description": "Number of results to return",
                    "default": 10,
                    "required": False,
                },
            },
            end=False,
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        import os

        return os.getenv("VECTOR_DB_TYPE", "").lower() == "qdrant"

    async def __call__(
        self,
        tree_data,
        inputs,
        base_lm,
        complex_lm,
        client_manager,
        **kwargs,
    ):
        from elysia.objects import Result, Status, Error

        # Parse inputs
        collection = inputs["collection"]
        raw_vec = inputs["query_vector"]
        limit = int(inputs.get("limit", 10) or 10)

        # Coerce vector
        try:
            if isinstance(raw_vec, str):
                parts = (
                    [p.strip() for chunk in raw_vec.split(",") for p in chunk.split()]
                    if "," in raw_vec
                    else raw_vec.split()
                )
                query_vector = [float(p) for p in parts if p != ""]
            elif isinstance(raw_vec, list):
                query_vector = [float(x) for x in raw_vec]
            else:
                yield Error("Unsupported query_vector input format.")
                return
        except Exception:
            yield Error("Query vector contains non-numeric values.")
            return

        if not query_vector:
            yield Error("Query vector is empty.")
            return

        # Load vector DB from env
        try:
            from elysia.api.vector_db import get_vector_db_from_env

            db = get_vector_db_from_env()
        except Exception as e:
            yield Error(f"Failed to initialize Qdrant client: {e}")
            return

        yield Status(
            f"Searching Qdrant collection '{collection}' with {len(query_vector)}-dim vector."
        )

        try:
            results = db.search_vectors(collection, query_vector, limit=limit)
        except Exception as e:
            yield Error(f"Qdrant search failed: {e}")
            return

        # Normalize minimal result shape
        normalized = [
            {
                "id": r.get("id"),
                "score": r.get("score"),
                "payload": r.get("payload", {}),
            }
            for r in results
        ]

        yield Result(
            objects=normalized,
            metadata={"collection": collection, "count": len(normalized)},
            name="qdrant_search_results",
            llm_message=(
                "Retrieved {count} nearest vectors from '{collection}' using Qdrant backend."
            ),
        )
