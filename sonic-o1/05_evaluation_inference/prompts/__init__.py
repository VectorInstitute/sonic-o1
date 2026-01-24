"""
prompts/__init__.py.

Prompt templates for tasks.
"""

from .t1_prompts import get_t1_empathy_prompt, get_t1_prompt
from .t2_prompts import get_t2_prompt
from .t3_prompts import get_t3_prompt


__all__ = ["get_t1_prompt", "get_t1_empathy_prompt", "get_t2_prompt", "get_t3_prompt"]
