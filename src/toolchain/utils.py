"""Utility functions for the toolchain."""
import aiofiles
import re
import json

from pathlib import Path

from typing import Literal, List, Union, Optional, TypeVar, Callable, Tuple, Any, Mapping
from typing_extensions import TypeAlias

from chainlit.element import ElementSize, ElementDisplay
from chainlit.message import Message

from toolchain.types import ReplicatePredictionStatus


def resize_images(message: Message, size: ElementSize = "small") -> Message:
    """Resize images in a message."""
    new_elements = []
    for el in message.elements:
        if el.type == "image":
            el.size = size
        new_elements.append(el)
    message.elements = new_elements
    return message


SafeInt = TypeVar('SafeInt', int, str, None)
from typing import Optional

def safe_int_parse(value, default: Optional[int]=None) -> int:
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
def safe_int_parse(value, default: Optional[SafeInt]=None) -> SafeInt:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_replicate_prediction_log(id: str, log_text: str, status: Optional[ReplicatePredictionStatus] = "starting"):
    """Extracts data segments from the log text using regex patterns.

    Args:
      id (str): The ID of the replicate prediction log.
      log_text (str): The log text to extract data from.
      status (Optional[ReplicatePredictionStatus], optional): The status of the replicate prediction. Defaults to "starting".

    Returns:
      dict: A dictionary containing the extracted data segments.
    """
    from toolchain.types import ReplicatePredictionStatus, ParsedReplicatePredictionLog
    
    data = ParsedReplicatePredictionLog(id=id, status=status)
    progress_key = "sampling_progress_pattern"
    img_dim_key = "image_dimensions_pattern"
    
    # Define regex patterns for segments of the log
    patterns = ParsedReplicatePredictionLog.from_dict({
        "seed_pattern": r"Using seed: (\d+)",
        img_dim_key: r"Using dimensions (\d+)x(\d+)",  # Capture both dimensions
        "model_loaded_pattern": r"Loaded model",
        "warning_pattern": r"WARNING: (.*?)\n##############################",  # Stop at the delimiter
        "sampler_pattern": r"Sampler: (.*?)\n",  # Stop at the end of the line
        "discretization_pattern": r"Discretization: (.*?)\n",  # Stop at the end of the line
        "guider_pattern": r"Guider: (.*?)\n",  # Stop at the end of the line
        "ffmpeg_output_pattern": r"(ffmpeg.*\n)+",
        progress_key: r"Sampling with (.*?) for (\d+) steps: +(\d+)%\|([█▏ ]+) \| (\d+)/(\d+)",  # Capture percentage directly
    })
    
    # Single line for loop to extract data from the log
    # _ = [data.update({key: match.group(1) if match.groups() else match.group(0) or ""}) for key, match in [(key, re.search(pattern, log_text, flags=re.DOTALL)) for key, pattern in patterns.items()]]
    
    # Extract the main data from the log
    for key, pattern in patterns.to_dict().items():
        if match := re.search(str(pattern), log_text, flags=re.DOTALL):
            primary_value = match.group(1) if match.groups() else match.group(0)
            if key == img_dim_key:
                primary_value = f"{primary_value}x{match.group(2)}"
            data.merge({key: primary_value})

    # Progress details
    progress_info: List[ParsedReplicatePredictionLog] = []
    # progress_pattern = r"Sampling with (.*?) for (\d+) steps: +(\d+)%\|([█▏ ]+) \| (\d+)/(\d+)"
    progress_pattern = r"Sampling with (.*?) for (\d+) steps: +(\d+)%\|([█▏]+)"
    
    # Extract the progress details
    for match in re.finditer(progress_pattern, log_text):
        sampler = match.group(1)
        total_steps = match.group(2)
        percent_complete = match.group(3)
        progress_bar = match.group(4)
        # current_step = match.group(5)
        # total_steps_again = match.group(6)

        progress_info.append(ParsedReplicatePredictionLog.from_dict({
            "sampler": sampler,
            "total_steps": safe_int_parse(total_steps, 26),
            "percent_complete": safe_int_parse(percent_complete, 0),
            "progress_bar": progress_bar,
            # "current_step": current_step,
            # "total_steps_again": total_steps_again
        }))
    data.merge({
        **(progress_info[-1].to_dict() if len(progress_info) > 0 else {}),
        progress_key: [progress.to_dict(infer_missing=False) for progress in progress_info],
    })
    return data


# _PROGRESS_BAR_SIMPLE_TEMPLATE_UNSAFE_HTML="""\
# <div class="progress-container" style="width:100%; background-color: #212121; border-radius: 5px; box-shadow: inset 2px 1px 2px 1px black;">
#   <div class="progress-bar" style="width:{PROGRESS}%; height: 30px; background-color: #f80061; text-align: center; line-height: 30px; color: #fff; text-shadow: 1px 1px 2px black; border-top-left-radius: 5px; border-bottom-left-radius: 5px; box-shadow: 3px 0px 10px 0px black;"><span style="margin-left: 10px;">{PROGRESS}%</span></div>
# </div>
# """


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

