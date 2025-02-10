"""
Prompt templates and action space definitions for UI-TARS agent.
"""
from typing import Dict

# Action space definitions with detailed descriptions
REFLECTION_ACTION_SPACE = """
click(start_box='[x1, y1, x2, y2]')
left_double(start_box='[x1, y1, x2, y2]')
right_single(start_box='[x1, y1, x2, y2]')
drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
"""

CALL_USER_REFLECTION_ACTION_SPACE = """
click(start_box='[x1, y1, x2, y2]')
left_double(start_box='[x1, y1, x2, y2]')
right_single(start_box='[x1, y1, x2, y2]')
drag(start_box='[x1, y1, x2, y2]', end_box='[x3, y3, x4, y4]')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='[x1, y1, x2, y2]', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.
"""

# Basic prompt template without thought process
NO_THOUGHT_PROMPT_0103 = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Action: ...
```
## Action Space
{action_space}
## User Instruction
{instruction}
"""

# Multi-step prompt template with thought process
MULTI_STEP_PROMPT_1229 = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
{action_space}

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

# Chinese multi-step prompt template with specific requirements
MULTI_STEP_ACTION_W_THOUGHT_TEMPLATE_M03_LONG_CH = """You need to output a thought with history summary and an action after each video keyframe, according to the user's instruction and history trajectories. 

**Things to Notice**:
- Use point to ground objects.
- Use Chinese.
- Output in ReACT format:

Thought: ...
Action: ...

- The action space is: {action_space}

**User Instruction**
{instruction}"""

# Mapping of prompt templates to their configurations
PROMPT_CONFIGS: Dict[str, Dict] = {
    "no_thought": {
        "template": NO_THOUGHT_PROMPT_0103,
        "action_space": CALL_USER_REFLECTION_ACTION_SPACE
    },
    "reflection": {
        "template": MULTI_STEP_PROMPT_1229,
        "action_space": REFLECTION_ACTION_SPACE
    },
    "chinese_long": {
        "template": MULTI_STEP_ACTION_W_THOUGHT_TEMPLATE_M03_LONG_CH,
        "action_space": REFLECTION_ACTION_SPACE
    }
} 
