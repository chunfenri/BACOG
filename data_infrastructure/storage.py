import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse


class StorageReader:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._memory_cache = {}               
        self._tool_cache = {}            
    
    def read(self, ptr: str) -> str:
        try:
            backend, key, span = self._parse_uri(ptr)
                                 
            if backend == 'fs':
                content = self._read_filesystem(key)
            elif backend == 'memory':
                content = self._read_memory(key)
            elif backend == 'tool':
                content = self._read_tool(key)
            else:
                return ""
                           
            if span and content:
                content = self._extract_span(content, span)
            
            return content
        
        except Exception as e:
            return ""
    
    def _parse_uri(self, ptr: str) -> Tuple[str, str, Optional[str]]:
        if '://' in ptr:
            backend, remaining = ptr.split('://', 1)
            pass
        else:
            pass
                               
        if '://' in ptr:
            backend, remaining = ptr.split('://', 1)
            if backend == 'store':
                                     
                if '/' in remaining:
                    backend, key = remaining.split('/', 1)
                else:
                    raise ValueError(f"Invalid store URI: {ptr}")
            else:
                key = remaining
        else:
                                                                                                              
            parts = ptr.split('/', 1)
            if len(parts) == 2 and not ptr.startswith('/'):
                 backend = parts[0]
                 key = parts[1]
            else:
                                                                            
                 backend = 'fs'
                 key = ptr
                        
        if '#' in key:
            key, span = key.split('#', 1)
        else:
            span = None
            
        return backend, key, span
    
    def _read_filesystem(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return ""
    
    def _read_memory(self, key: str) -> str:
        return self._memory_cache.get(key, "")
    
    def _read_tool(self, key: str) -> str:
        return self._tool_cache.get(key, "")
    
    def _extract_span(self, content: str, span: str) -> str:
        match = re.match(r'L(\d+)-(\d+)', span)
        if not match:
            return content
        
        start_line = int(match.group(1))
        end_line = int(match.group(2))
        
        lines = content.split('\n')
                            
        selected_lines = lines[start_line-1:end_line]
        
        return '\n'.join(selected_lines)
    
    def add_to_memory(self, key: str, content: str):
        self._memory_cache[key] = content
    
    def add_to_tool_cache(self, key: str, content: str):
        self._tool_cache[key] = content
    
    def clear_cache(self):
        self._memory_cache.clear()
        self._tool_cache.clear()

