"""
Template rendering utilities for LLM prompts.
Provides consistent handling of system/user prompt separation.
"""
import os
import jinja2
from typing import Tuple, Optional

# Setup Jinja2 environment
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR))


def render_template(template_name: str, **kwargs) -> str:
    """
    Render a template and return the full content.
    
    For backward compatibility with templates that don't use system/user separation.
    
    Args:
        template_name: Path to template file relative to prompts/
        **kwargs: Variables to pass to the template
    
    Returns:
        Rendered template string
    """
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


def render_template_with_prompts(template_name: str, **kwargs) -> Tuple[Optional[str], str]:
    """
    Render a template and extract system_prompt and user_prompt.
    
    Templates should define:
        {% set system_prompt %}...{% endset %}
        {% set user_prompt %}...{% endset %}
    
    If not defined, returns (None, full_content) for backwards compatibility.
    
    Args:
        template_name: Path to template file relative to prompts/
        **kwargs: Variables to pass to the template
    
    Returns:
        Tuple of (system_prompt, user_prompt)
        - system_prompt may be None if not defined in template
        - user_prompt contains the main prompt content
    """
    template = jinja_env.get_template(template_name)
    
    # Create a context with the variables
    context = template.new_context(kwargs)
    
    # Render and execute to populate variables
    # This captures the output in a list, but we mainly care about context.vars
    content = "".join(template.root_render_func(context))
    
    # Extract system_prompt and user_prompt from context
    system_prompt = context.vars.get('system_prompt')
    user_prompt = context.vars.get('user_prompt')
    
    if system_prompt:
        system_prompt = str(system_prompt).strip()
    
    if user_prompt:
        user_prompt = str(user_prompt).strip()
    else:
        # Fallback: if no user_prompt defined, use full rendered content
        user_prompt = content.strip()
    
    return system_prompt, user_prompt
