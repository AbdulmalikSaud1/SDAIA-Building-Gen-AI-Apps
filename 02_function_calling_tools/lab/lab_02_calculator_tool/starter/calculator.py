"""
Lab 2: Calculator Tool — Schema, Execution, and Resilience
============================================================
Implement a production-ready calculator tool with:
1. A JSON Schema definition for the LLM
2. An execute_calculation function with structured error returns
3. A resilient_api_call decorator for external API calls
"""

import json
import logging
import functools
from typing import Dict, Any

logger = logging.getLogger(__name__)

# =============================================================================
# Step 1: Define the Tool Schema
# =============================================================================
# This schema tells the LLM what the tool does and how to call it.
# The description is critical — it's a prompt to the model.

CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_calculation",
        "description": (
            "Executes a basic arithmetic or exponentiation operation. "
            "Use for any math in user questions: percentages, growth rates, "
            "compound interest, splits, or simple arithmetic. "
            "Example: For 'What is 15% of 200?', use operation='multiply', "
            "operand_a=200, operand_b=0.15."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "pow"],
                    "description": "The arithmetic operation. 'pow' calculates operand_a to the power of operand_b."
                },
                "operand_a": {
                    "type": "number",
                    "description": "The first operand (base for 'pow')."
                },
                "operand_b": {
                    "type": "number",
                    "description": "The second operand (exponent for 'pow'). For division, this is the divisor."
                }
            },
            "required": ["operation", "operand_a", "operand_b"]
        }
    }
}


# =============================================================================
# Step 2: Implement the Execution Function
# =============================================================================

def execute_calculation(operation: str, operand_a: float, operand_b: float) -> Dict[str, Any]:
    """
    Performs the calculation and returns a structured result.

    Must ALWAYS return a dict with keys: success, result, error.
    Never raise an uncaught exception.

    Args:
        operation: One of "add", "subtract", "multiply", "divide", "pow"
        operand_a: The first operand
        operand_b: The second operand

    Returns:
        {"success": True/False, "result": <number or None>, "error": <str or None>}
    """
    logger.info(f"Executing calculation: {operand_a} {operation} {operand_b}")

    result = None
    error = None

    try:
        # Implement the operation logic
        if operation == "add":
            result = operand_a + operand_b
        elif operation == "subtract":
            result = operand_a - operand_b
        elif operation == "multiply":
            result = operand_a * operand_b
        elif operation == "divide":
            if operand_b == 0:
                error = "Cannot divide by zero"
            else:
                result = operand_a / operand_b
        elif operation == "pow":
            result = operand_a ** operand_b
        else:
            error = f"Unsupported operation: {operation}"
    except Exception as e:
        error = f"Calculation error: {str(e)}"

    # Build the response
    if error:
        logger.warning(f"Calculation failed: {error}")
        return {"success": False, "result": None, "error": error}
    else:
        logger.info(f"Calculation successful: {result}")
        return {"success": True, "result": result, "error": None}


# =============================================================================
# Step 3: Build the Resilient API Call Decorator
# =============================================================================

def resilient_api_call(max_retries: int = 2, timeout_seconds: int = 10):
    """
    A decorator that adds retries with exponential backoff to any function
    that makes an external API call.

    Usage:
        @resilient_api_call(max_retries=2, timeout_seconds=5)
        def get_stock_price(ticker: str):
            ...

    Args:
        max_retries: Number of retry attempts after the initial call
        timeout_seconds: Timeout in seconds for external requests
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: Implement retry logic
            # Option A (recommended): Use tenacity library
            #   from tenacity import retry, stop_after_attempt, wait_exponential
            #   Wrap the function call with retry behavior
            #
            # Option B (manual): Use a for loop with time.sleep
            #   for attempt in range(max_retries + 1):
            #       try: return func(*args, **kwargs)
            #       except: time.sleep(2 ** attempt)
            #
            # On final failure, return:
            #   {"success": False, "error": "Service unavailable after retries"}
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return {"success": False, "error": f"Service error: {str(e)}"}
        return wrapper
    return decorator


# =============================================================================
# Helper: Tool Dispatch
# =============================================================================

def get_tool_schemas() -> list:
    """Returns the list of tool schemas for the LLM API."""
    return [CALCULATOR_SCHEMA]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Simple dispatcher — routes tool calls to the right function."""
    if tool_name == "execute_calculation":
        return execute_calculation(**arguments)
    else:
        return {"success": False, "error": f"Unknown tool: {tool_name}", "result": None}


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the calculator
    print(execute_calculation("add", 10, 5))
    print(execute_calculation("divide", 10, 0))
    print(execute_calculation("pow", 2, 10))
