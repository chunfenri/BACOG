import json
from typing import List, Tuple, Dict


class JsonTemplate:
    def build(self, 
             task_state: dict,
             compressed_clues: List[Tuple[str, str]],
             schema: dict,
             b_out: int) -> str:
        state_section = self._build_state_section(task_state)
        
        inputs_section = self._build_inputs_section(compressed_clues)
        
        constraints_section = self._build_constraints_section(schema, b_out)
        
        prompt = f"""{state_section}

{inputs_section}

{constraints_section}

# Your Response (JSON ONLY):
"""
        return prompt
    
    def _build_state_section(self, task_state: dict) -> str:
        goal = task_state.get('goal', 'Complete the task')
        step = task_state.get('step', 1)
        plan = task_state.get('plan', [])
        total_steps = len(plan) if plan else 1
        must_have = task_state.get('success', {}).get('must_have', [])
        
        section = f"""# Task State
Goal: {goal}"""

                                                                
        global_goal = task_state.get('global_goal')
        if global_goal and global_goal != goal:
            section += f"\nGlobal Goal: {global_goal}"

        section += f"\nStep: {step}/{total_steps}"
        
        if must_have:
            section += f"\nMust-have fields: {', '.join(must_have)}"
        
        return section
    
    def _build_inputs_section(self, compressed_clues: List[Tuple[str, str]]) -> str:
        if not compressed_clues:
            return "# Clues\n(No clues available)"
        clue_lines = []
        for i, (clue_id, compressed_text) in enumerate(compressed_clues, 1):
            ct = (compressed_text or "").strip()
            clue_lines.append(f"{i}. [{clue_id}] {ct}")
        
        section = f"""# Clues
{chr(10).join(clue_lines)}"""
        
        return section
    
    def _build_constraints_section(self, schema: dict, b_out: int) -> str:
        is_search_step = 'rationalized_query' in str(schema.get('properties', {})) or\
                        'rationalized_query' in str(schema.get('required', []))
        
        schema_str = json.dumps(schema, ensure_ascii=False)
        required_keys = schema.get("required", []) if isinstance(schema, dict) else []
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}

        def _example_value(prop_schema: dict):
            t = (prop_schema or {}).get("type")
            if t == "string":
                return "<string>"
            if t == "number":
                return 0.0
            if t == "integer":
                return 0
            if t == "boolean":
                return False
            if t == "array":
                return []
            if t == "object":
                return {}
            return "<value>"

        example_obj = {}
        if isinstance(required_keys, list) and required_keys:
            for k in required_keys:
                if isinstance(k, str):
                    example_obj[k] = _example_value(props.get(k, {}) if isinstance(props, dict) else {})
        else:
            if isinstance(props, dict) and props:
                first_k = next(iter(props.keys()))
                example_obj[first_k] = _example_value(props.get(first_k, {}))
        example_str = json.dumps(example_obj, ensure_ascii=False)
        
        req_str = ", ".join([k for k in required_keys if isinstance(k, str)]) if isinstance(required_keys, list) else ""
        req_line = f"Required keys: {req_str}" if req_str else "Required keys: (see schema)"

        section = f"""# Output (JSON only)
Return exactly ONE JSON object.
Do NOT output schema/meta keys: type, properties, required.
{req_line}

Schema (reference only):
{schema_str}

Example (shape only):
{example_str}"""
        
        if is_search_step:
            section += """

SEARCH step:
- Output ONE short fact sentence that helps answer the Goal.
- Output JSON only (no extra text)."""
        else:
            section += """

QA step:
- Answer the Goal directly and concisely in the required JSON.
- Yes/No: output exactly \"yes\" or \"no\".
- Numbers/dates: output the number/date string. No explanations/reasoning text."""
        
        return section
