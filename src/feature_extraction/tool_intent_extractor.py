class ToolIntentExtractor:
    def __init__(self):
        self.tool_keys = ["tool_calls", "function_call", "tool_use"]

    def extract_tool_flag(self, raw_response: dict) -> bool:
        if not raw_response or not raw_response.get("choices"):
            return False
        
        first_choice = raw_response["choices"][0]
        message = first_choice.get("message", {})
        
        for key in self.tool_keys:
            if message.get(key) is not None:
                return True
        return False







