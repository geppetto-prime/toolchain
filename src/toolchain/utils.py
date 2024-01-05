"""Utility functions for the toolchain."""
import aiofiles
import json

from pathlib import Path

from typing import Literal, List, Union, Optional, TypeVar, Callable

from chainlit.element import ElementSize
from chainlit.message import Message



def resize_images(message: Message, size: ElementSize = "small") -> Message:
    """Resize images in a message."""
    new_elements = []
    for el in message.elements:
        if el.type == "image":
            el.size = size
        new_elements.append(el)
    message.elements = new_elements
    return message
    
from typing import List

def safe_name_parse(name: str, allowed: List[str] = ['-', '_', '(', ')', '[', ']', '{', '}']) -> str:
    """
    Safely parse a name by removing spaces, commas, and characters not in the allowed list.

    Args:
        name (str): The name to be parsed.
        allowed (List[str], optional): A list of allowed characters. Defaults to ['-', '_', '(', ')', '[', ']', '{', '}'].

    Returns:
        str: The parsed name.
    """
    trimmed_name = ""
    for char in name.replace(" ", "_").replace(",", "").strip():
        if char.isalnum() or char in allowed:
            trimmed_name += char
    return trimmed_name


SafeInt = TypeVar('SafeInt', int, str, None)

def safe_int_parse(value, default: Optional[SafeInt]=None) -> SafeInt:
    """
    Safely parses the given value into an integer.

    Args:
        value: The value to be parsed.
        default: The default value to be returned if parsing fails. Defaults to None.

    Returns:
        The parsed integer value if successful, otherwise the default value.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


_PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML_PART_PROGRESS_LABEL = """\
<span class="progress-label" style="">{TEXT}</span>
"""

_PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML_PART_PROGRESS_HEADER = """\
<span class="progress-header" style="">{TEXT}</span>
"""

_PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML = """\
<div class="progress-root">
  {PROGRESS_HEADER}
  <div class="progress-container" style="">
    <div class="progress-bar" style="width:{PROGRESS_PERCENTAGE}%;">
      {PROGRESS_LABEL}
    </div>
  </div>
</div>
"""


def progress_bar_simple(
    progress_percentage: int = 0,
    progress_label_fn: Optional[Callable[[int], str]] = lambda progress: f"{progress}%",
    progress_header_fn: Optional[Callable[[int], str]] = None,
) -> str:
    """Returns a simple progress bar.

    Args:
        progress_percentage (int): The percentage completion of the task (0-100).
        progress_label_fn (Optional[Callable[[int], str]], optional): A function that generates the label for the progress bar based on the progress percentage. Defaults to lambda progress: f"{progress}%".
        progress_header_fn (Optional[Callable[[int], str]], optional): A function that generates the header for the progress bar based on the progress percentage. Defaults to None.

    Returns:
        str: The HTML representation of the progress bar.
    """
    progress_percentage = safe_int_parse(progress_percentage, 0)
    # Ensure the percentage is within 0-100 range
    progress_percentage = max(0, min(100, progress_percentage))

    # Generate label and header if functions are provided
    progress_label = progress_label_fn(progress_percentage) if progress_label_fn else ""
    progress_header = progress_header_fn(progress_percentage) if progress_header_fn else ""

    html_output = _PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML.format(
        PROGRESS_PERCENTAGE=progress_percentage,
        PROGRESS_LABEL=_PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML_PART_PROGRESS_LABEL.format(TEXT=progress_label),
        PROGRESS_HEADER=_PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML_PART_PROGRESS_HEADER.format(TEXT=progress_header),
    )
    return html_output


async def progress_bar_simple_test(sleepfn: Callable[[int], None], seconds: int = 5, only_percent: bool = False):
    """
    Tests the progress bar.

    Args:
        sleepfn (Callable[[int], None]): A function that sleeps for a given number of seconds.
        seconds (int, optional): The total number of seconds for the progress bar. Defaults to 5.
        only_percent (bool, optional): If True, only yields the calculated percentage. If False, yields the progress bar string. Defaults to False.

    Yields:
        int or str: The calculated percentage or the progress bar string.

    """
    for i in range(seconds):
        calc_percent = int((i / seconds) * 100)
        if only_percent:
            yield calc_percent
        else:
            yield progress_bar_simple(calc_percent)
        await sleepfn(1)
    if only_percent:
        yield 100
    else:
        yield progress_bar_simple(100)


def progress_bar_simple_markdown(
    progress_percentage: int = 0,
    style: Literal["default", "variant-1", "variant-2"] = "default",
):
    """Returns a simple progress bar. Needs work.
    `progress_percentage` is the percentage completion of the task (0-100)."""
    
    PROGRESS_BAR_MARKDOWN_TEMPLATE = """**Progress:** *{LSIDE}{TRANSITION}{RSIDE}* {PROGRESS}% {STATUS}"""
    
    # Ensure the percentage is within 0-100 range
    progress_percentage = max(0, min(100, progress_percentage))
    
    transition_0 = "█"
    transition_a = "█▓▒░"
    transition_b = "░▒▓█"
    progress = progress_percentage // 2
    transition = transition_0
    if style == "variant-a":
        transition = transition_a
    elif style == "variant-b":
        transition = transition_b

    # style_01 = f"**Progress:** *{'░' * progress}{transition_a}{'░' * (50 - progress)}* {progress * 2}%"
    # style_02 = f"**Progress:** *{'░' * progress}{transition_b}{'░' * (50 - progress)}* {progress * 2}%"
    # style_03 = f"**Progress:** *{'█' * progress}{transition_0}{'░' * (50 - progress)}* {progress * 2}%"
    # markdown_template = lambda progress, transition=transition_0: f"**Progress:** *{'░' * progress}{transition}{'░' * (50 - progress)}* {progress * 2}%"
    markdown_template = lambda progress, transition=transition_0: f"**Progress:** {transition[:1] * progress}{transition}{transition[-1] * (50 - progress)} {progress_percentage}%"
    return markdown_template(progress, transition=transition)


async def append_to_json_log(data, log_path: Union[str, Path]):
    """
    Appends data to a JSON log file.

    Args:
        data: The data to append to the log file.
        log_path (Union[str, Path]): The path to the log file.

    Raises:
        None

    Returns:
        None
    """
    log_path = Path(log_path)
    if log_path.suffix != ".json":
        print(f"Notice: `append_to_json_log` is intended to be passed a .json file but was passed a {log_path.suffix} file.")
    try:
        # Read existing data
        if not log_path.exists():
            log_path.touch(mode=0o777, exist_ok=True)
        async with aiofiles.open(log_path, mode='r') as file:
            existing_data = json.loads(await file.read())
    
        # Append new data
        existing_data.append(data)

        # Write back to file
        async with aiofiles.open(log_path, mode='w') as file:
            await file.write(json.dumps(existing_data, indent=4))
    except Exception as e:
        pass

