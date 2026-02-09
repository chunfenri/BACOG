import os
import time
import json
from typing import Optional, Dict, Any

from bacog.configs.system_config import config


class LLMClient:
    def __init__(self, 
                 model: str = "gemini-3-flash-preview-all", 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.model = model
        cfg_llm = getattr(config, "llm", None)
        if api_key is None and cfg_llm is not None:
            api_key = getattr(cfg_llm, "API_KEY", None) or None
        if base_url is None and cfg_llm is not None:
            base_url = getattr(cfg_llm, "BASE_URL", None) or None
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            pass
    
    def call(self, 
             prompt: str, 
             max_tokens: Optional[int] = None,
             temperature: float = 0.0,
             response_format: Optional[Dict[str, str]] = None,
             stop: Optional[list] = None) -> str:
        if not self.api_key:
            raise RuntimeError("API key not configured")
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_api(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    stop=stop
                )
                return response
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {str(e)}")
    
    def _call_api(self,
                  prompt: str,
                  max_tokens: Optional[int],
                  temperature: float,
                  response_format: Optional[Dict[str, str]],
                  stop: Optional[list]) -> str:
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        client = openai.OpenAI(**client_kwargs)
        
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        if response_format:
            request_params["response_format"] = response_format
        
        if stop:
            request_params["stop"] = stop
        
        response = client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content
        
        if content is None:
            raise RuntimeError("LLM returned empty response")
        
        return content.strip()

    def call_with_usage(self, 
             prompt: str, 
             max_tokens: Optional[int] = None,
             temperature: float = 0.0,
             response_format: Optional[Dict[str, str]] = None,
             stop: Optional[list] = None) -> tuple[str, Dict[str, int]]:
        if not self.api_key:
            raise RuntimeError("API key not configured")
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_api_full(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    stop=stop
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty response")
                    
                usage_dict = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    if isinstance(usage, dict):
                        usage_dict = {
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0)
                        }
                    else:
                        usage_dict = {
                            'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                            'completion_tokens': getattr(usage, 'completion_tokens', 0),
                            'total_tokens': getattr(usage, 'total_tokens', 0)
                        }
                
                return content.strip(), usage_dict
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {str(e)}")

    def _call_api_full(self,
                  prompt: str,
                  max_tokens: Optional[int],
                  temperature: float,
                  response_format: Optional[Dict[str, str]],
                  stop: Optional[list]):
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        client = openai.OpenAI(**client_kwargs)
        
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        if response_format:
            request_params["response_format"] = response_format
        
        if stop:
            request_params["stop"] = stop
        
        return client.chat.completions.create(**request_params)
    
    def call_with_json(self,
                      prompt: str,
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.0) -> Dict[str, Any]:
        response_text = None
        
        try:
            response_text = self.call(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        except RuntimeError as e:
            if "response_format" in str(e).lower():
                response_text = self.call(
                    prompt=prompt + "\n\nOutput ONLY valid JSON, no other text.",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=None
                )
            else:
                raise
        
        return self._parse_json_response(response_text)

    def call_with_json_with_usage(
                      self,
                      prompt: str,
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.0) -> tuple[Dict[str, Any], Dict[str, int]]:
        response_text = None
        usage_dict = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        try:
            response_text, usage_dict = self.call_with_usage(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        except RuntimeError as e:
            if "response_format" in str(e).lower():
                response_text, usage_dict = self.call_with_usage(
                    prompt=prompt + "\n\nOutput ONLY valid JSON, no other text.",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=None
                )
            else:
                raise

        return self._parse_json_response(response_text), usage_dict
    
    def estimate_cost(self, 
                     input_tokens: int, 
                     output_tokens: int) -> float:
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gemini-3-flash-preview-all": {"input": 0.0005, "output": 0.003},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        model_pricing = pricing.get(self.model, pricing["gpt-4o-mini"])
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost

    def _clean_json_string(self, text: str) -> str:
        import re
        
        text = text.strip()
        
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        if text and not (text.startswith('[') or text.startswith('{')):
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                text = match.group(0)
            else:
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    text = match.group(0)
            
        return text.strip()

    def _parse_json_response(self, raw_text: str) -> Dict[str, Any]:
        if raw_text is None:
            raise RuntimeError("Invalid JSON response: empty content")

        cleaned = self._clean_json_string(str(raw_text))

        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"_list": parsed}

        text = str(raw_text)
        extracted = self._extract_first_balanced_json(text)
        if extracted is not None:
            extracted_clean = self._clean_json_string(extracted)
            try:
                parsed2 = json.loads(extracted_clean)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON response: {str(e)}")

            if isinstance(parsed2, dict):
                return parsed2
            if isinstance(parsed2, list):
                return {"_list": parsed2}

        raise RuntimeError("Invalid JSON response: cannot find valid JSON object/array in model output")

    def _extract_first_balanced_json(self, text: str) -> Optional[str]:
        if not text:
            return None

        i_obj = text.find("{")
        i_arr = text.find("[")
        starts = [i for i in [i_obj, i_arr] if i != -1]
        if not starts:
            return None
        start = min(starts)
        opener = text[start]
        closer = "}" if opener == "{" else "]"

        stack = []
        in_str = False
        esc = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == "\"":
                    in_str = False
                continue

            if ch == "\"":
                in_str = True
                continue

            if ch == "{" or ch == "[":
                stack.append(ch)
            elif ch == "}" or ch == "]":
                if not stack:
                    continue
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        return text[start : idx + 1]
                else:
                    continue

        return None

