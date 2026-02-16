"""
Lab 2: Agent Core — OpenAI Integration with Tool Calling
==========================================================
Wire the calculator tool into OpenAI's Chat Completions API.
Implements the two-call pattern:
  1. Send messages + tool schemas → get tool_calls
  2. Execute tools → send results back → get final answer
"""

import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

from calculator import get_tool_schemas, execute_tool

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_ai_response_with_tools(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Sends messages to OpenAI, handling tool calls if returned.

    The two-call pattern:
    1. First API call: send messages + tool schemas
    2. If tool_calls in response: parse, execute, append results
    3. Second API call: send updated messages → get final answer

    Args:
        messages: The conversation history
        model: The OpenAI model to use

    Returns:
        {"response_text": str, "tool_results": list}
    """

    # --- First API Call ---
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tool_schemas(),   # Inject tool schemas
            tool_choice="auto",         # Let the model decide
            temperature=0.1             # Low temp for deterministic tool use
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return {
            "response_text": "I'm having trouble connecting. Please try again.",
            "tool_results": []
        }

    response_message = response.choices[0].message
    tool_results = []

    # --- Check for Tool Calls ---
    if response_message.tool_calls:
        logger.info(f"Model initiated {len(response_message.tool_calls)} tool call(s).")

        # Append the assistant's message (with tool_calls) to messages
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": response_message.tool_calls
        })
        
        # Loop through each tool_call
        for tool_call in response_message.tool_calls:
            # 1. Extract tool_name
            tool_name = tool_call.function.name
            
            # 2. Parse arguments
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {e}")
                arguments = {}
            
            # 3. Execute the tool
            result = execute_tool(tool_name, arguments)
            logger.info(f"Tool {tool_name} returned: {result}")
            
            # 4. Append the tool result message to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            
            # 5. Collect results
            tool_results.append(result)

        # Make the second API call with updated messages
        second_response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1
        )
        response_text = second_response.choices[0].message.content
    else:
        # No tool calls — direct text response
        response_text = response_message.content

    return {
        "response_text": response_text,
        "tool_results": tool_results
    }


# =============================================================================
# Interactive Chat Loop
# =============================================================================
if __name__ == "__main__":
    print("Calculator Agent (type 'quit' to exit)")
    print("=" * 40)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a calculator tool. Use it for any mathematical calculations."}
    ]

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        result = get_ai_response_with_tools(messages)

        print(f"\nAssistant: {result['response_text']}")

        if result["tool_results"]:
            print(f"  [Tools used: {len(result['tool_results'])}]")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": result["response_text"]})
