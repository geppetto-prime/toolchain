
import re
import dataclasses

from typing import (
    Callable, List, Dict, Any, Optional, Literal, Tuple, Type, TypeVar, Union, Mapping,
    BinaryIO, TextIO, Awaitable,
)
from typing_extensions import TypeAlias

from pathlib import Path

from pydantic import BaseModel, Field, root_validator, ConfigDict
from pydantic.dataclasses import dataclass

from chainlit.input_widget import InputWidget, Select, Slider, NumberInput, TextInput, Switch, Tags
from chainlit.types import InputWidgetType

from toolchain.models import DataClassJsonMixinPro
from toolchain.utils import safe_int_parse
from toolchain.links.replicate.types import (
    PredictionStatus, StableDiffusionVideo_SizingStrategy, StableDiffusionVideo_VideoLength,
    PlaygroundV2_1024pxAesthetic_Scheduler,
)

from replicate.prediction import Prediction



class ReplicateModel:
    model_identifier: str
    
    def __init__(self, model_identifier: str) -> None:
        self.model_identifier = model_identifier

    @property
    def model(self) -> str:
        """Get the model name from the replicate_model attribute.

        Returns:
            str: The model name."""
        return self.model_identifier.split(":").pop(0)

    @property
    def version(self) -> str:
        """Get the version from the replicate_model attribute.

        Returns:
            str: The version."""
        return self.model_identifier.split(":").pop()




ParsedPredictionLogType = TypeVar('ParsedPredictionLogType', bound="ParsedPredictionLog")


@dataclass
class ParsedPredictionLog(DataClassJsonMixinPro):
    id: Optional[str] = dataclasses.field(default=None)
    status: Optional[PredictionStatus] = dataclasses.field(default=None)
    seed_pattern: Optional[str] = dataclasses.field(default=None)
    image_dimensions_pattern: Optional[str] = dataclasses.field(default=None)
    model_loaded_pattern: Optional[str] = dataclasses.field(default=None)
    warning_pattern: Optional[str] = dataclasses.field(default=None)
    sampler_pattern: Optional[str] = dataclasses.field(default=None)
    discretization_pattern: Optional[str] = dataclasses.field(default=None)
    guider_pattern: Optional[str] = dataclasses.field(default=None)
    ffmpeg_output_pattern: Optional[str] = dataclasses.field(default=None)
    sampling_progress_pattern: Optional[str] = dataclasses.field(default=None)
    sampler: Optional[str] = dataclasses.field(default=None)
    total_steps: Optional[int] = dataclasses.field(default=None)
    percent_complete: Optional[int] = dataclasses.field(default=None)
    progress_bar: Optional[str] = dataclasses.field(default=None)
    current_step: Optional[Union[int, str]] = dataclasses.field(default=None)
    prediction: Optional[Prediction] = dataclasses.field(default=None)
    
    @classmethod
    def parse(
        cls: Type[ParsedPredictionLogType],
        prediction: Prediction,
    ):
        """Extracts data segments from the log text using regex patterns.

        Args:
          id (str): The ID of the replicate prediction log.
          log_text (str): The log text to extract data from.
          status (Optional[PredictionStatus], optional): The status of the replicate prediction. Defaults to None.

        Returns:
          dict: A dictionary containing the extracted data segments.
        """
        percent_complete: int = 0
        current_step: int = 0
        total_steps: int = 0
        if prediction.progress:
            percent_complete = safe_int_parse(prediction.progress.percentage * 100, 0)
            current_step = prediction.progress.current
            total_steps = prediction.progress.total
        if prediction.status == "succeeded":
            percent_complete = 100
        log_text = prediction.logs or ""
        
        data = cls(id=prediction.id, status=prediction.status, percent_complete=percent_complete, current_step=current_step, total_steps=total_steps)
        
        progress_key = "sampling_progress_pattern"
        img_dim_key = "image_dimensions_pattern"
        
        # Define regex patterns for segments of the log
        patterns = cls.from_dict({
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
        progress_info: List[ParsedPredictionLogType] = []
        # progress_pattern = r"Sampling with (.*?) for (\d+) steps: +(\d+)%\|([█▏ ]+) \| (\d+)/(\d+)"
        progress_pattern = r"Sampling with (.*?) for (\d+) steps: +(\d+)%\|([█▏]+)"
        
        # Extract the progress details
        for match in re.finditer(progress_pattern, log_text):
            sampler = match.group(1)
            total_steps = match.group(2)
            percent_complete = match.group(3)
            progress_bar = match.group(4)
            # current_step = match.group(5)
            
            progress_segment = cls.from_dict({
                "sampler": sampler,
                "total_steps": safe_int_parse(total_steps, 26),
                "percent_complete": safe_int_parse(percent_complete, 0),
                "progress_bar": progress_bar,
                # "current_step": current_step,
            })
            progress_info.append(progress_segment)
        
        data.merge({
            **(progress_info[-1].to_dict() if len(progress_info) > 0 else {}),
            progress_key: [progress.to_dict(infer_missing=False) for progress in progress_info],
        })
        return data
    


T = TypeVar("T")


def field_default(
    default: Optional[T] = dataclasses.MISSING,
    default_factory: Callable[[], T] = dataclasses.MISSING,
    description: str = None,
    tooltip: str = None,
    widget_type: InputWidgetType = None,
    values: List[T] = None,
    max: Optional[T] = None,
    min: Optional[T] = None,
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = None,
):
    data_field = dataclasses.Field(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata={
            **(metadata or {}),
            "initial": default,
            # "initial_factory": default_factory,
            "description": description or tooltip,
            "tooltip": tooltip or description,
            **({"values": values} if values is not None else {}),
            "type": widget_type,
            **({"max": max} if max is not None else {}),
            **({"min": min} if min is not None else {}),
        },
        kw_only=kw_only,
    )
    return data_field

def field_default_factory(
    # default: Optional[T] = dataclasses.MISSING,
    default_factory: Callable[[], T] = dataclasses.MISSING,
    description: str = None,
    tooltip: str = None,
    widget_type: InputWidgetType = None,
    values: List[T] = None,
    max: Optional[T] = None,
    min: Optional[T] = None,
    *,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    metadata: Mapping[Any, Any] | None = None,
    kw_only: bool = None,
):
    data_field = dataclasses.field(
        default_factory=default_factory,
        metadata={
            **(metadata or {}),
            # "initial": default,
            "initial_factory": default_factory,
            "description": description or tooltip,
            "tooltip": tooltip or description,
            **({"values": values} if values is not None else {}),
            "type": widget_type,
            **({"max": max} if max is not None else {}),
            **({"min": min} if min is not None else {}),
        },
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        kw_only=kw_only,
    )
    return data_field

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class PlaygroundV2_1024pxAestheticParams(DataClassJsonMixinPro):
    """
    Represents the parameters for generating a Playground V2 1024px Aesthetic image.
    """
    prompt: str = field_default(default="")
    negative_prompt: str = field_default(default="")
    width: int = field_default(default=1024,
        description="The width of the image to generate.",
    )
    height: int = field_default(default=1024,
        description="The height of the image to generate.",
    )
    scheduler: PlaygroundV2_1024pxAesthetic_Scheduler = field_default(default="K_EULER_ANCESTRAL",
        description="The scheduler to use.",
        widget_type="select",
        values=["DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER", "PNDM", "DPM++_SDE_Karras"],
    )
    num_inference_steps: int = field_default(default=50,
        description="Number of denoising steps.",
        widget_type="slider",
        min=1,
        max=500,
    )
    guidance_scale: int = field_default(default=3,
        description="Scale for classifier-free guidance.",
        widget_type="slider",
        min=1,
        max=50,
    )
    apply_watermark: bool = field_default(default=False,
        description=(
            "Applies a watermark to enable determining if an image is generated "
            "in downstream applications. If you have other provisions for generating "
            "or deploying images safely, you can use this to disable watermarking."
        )
    )
    disable_safety_checker: bool = field_default(default=True,
        description="Disable safety checker.",
        widget_type="switch",
        tooltip="Disable safety checker for generated images.",
    )




@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StableDiffusionVideoParams(DataClassJsonMixinPro):
    """
    Represents the parameters for generating a stable diffusion video.

    Attributes:
        video_length (StableDiffusionVideo_VideoLength): The length of the video to generate.
        sizing_strategy (StableDiffusionVideo_SizingStrategy): The sizing strategy to use when generating the video.
        frames_per_second (int): The number of frames per second to use in the video.
        motion_bucket_id (int): The motion bucket id to use when generating the video.
        cond_aug (float): The conditional augmentation (cond aug) to use when generating the video.
        decoding_t (int): The decoding t to use when generating the video.
        seed (Optional[int]): The seed to use when generating the video.
        input_image (Optional[Union[str, TextIO, bytes, BinaryIO, Any]]): The input image to use when generating the video.
    """
    video_length: StableDiffusionVideo_VideoLength = dataclasses.field(
        default="25_frames_with_svd_xt",
        metadata={
            "description": "The length of the video to generate.",
            "tooltip": "The length of the video to generate.",
            "values": ["14_frames_with_svd", "25_frames_with_svd_xt"],
            "type": "select",
        }
    )
    sizing_strategy: StableDiffusionVideo_SizingStrategy = dataclasses.field(
        default="maintain_aspect_ratio",
        metadata={
            "description": "The sizing strategy to use when generating the video.",
            "tooltip": "Decide how to resize the input image.",
            "values": ["maintain_aspect_ratio", "crop_to_16_9", "use_image_dimensions"],
            "type": "select",
        }
    )
    frames_per_second: int = dataclasses.field(
        default=6,
        metadata={
            "description": "The number of frames per second to use in the video.",
            "tooltip": "The number of frames per second to use in the video.",
            "ge": 1,
            "le": 30,
            "type": "slider",
        }
    )
    motion_bucket_id: int = dataclasses.field(
        default=127,
        metadata={
            "description": "The motion bucket id to use when generating the video.",
            "tooltip": "Increase overall motion in the generated video.",
            "ge": 1,
            "le": 255,
            "type": "slider",
        }
    )
    cond_aug: float = dataclasses.field(
        default=0.02,
        metadata={
            "description": "The conditional augmentation (cond aug) to use when generating the video.",
            "tooltip": "Amount of noise to add to input image.",
            "type": "numberinput",
        }
    )
    decoding_t: int = dataclasses.field(
        default=14,
        metadata={
            "description": "The decoding t to use when generating the video.",
            "tooltip": "Number of frames to decode at a time.",
            "type": "numberinput",
        }
    )
    seed: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "description": "The seed to use when generating the video.",
            "tooltip": "Random seed. Leave blank to randomize the seed.",
            "type": "numberinput",
        }
    )
    input_image: Optional[Union[str, TextIO, bytes, BinaryIO, Any]] = dataclasses.field(
        default=None,
        metadata={
            "description": "The input image to use when generating the video.",
        },
    )
    # def input_widgets(self) -> List[InputWidget]:
    #     metadatas: Mapping[str, dict] = self.fields_metadata()
    #     widgets: List[InputWidget] = []
    #     for key in metadatas.keys():
    #         print(key, end=": ")
    #         metadata = metadatas.get(key, {})
    #         description = metadata.get('description', '')
    #         tooltip = metadata.get('tooltip', '')
    #         widget_type: InputWidgetType = metadata.get('type', '')
    #         values = metadata.get('values', [])
    #         initial = self.field_lookup(key).default
    #         max = metadata.get('ge', None)
    #         min = metadata.get('le', None)
    #         print(initial, end="; ")
    #         if widget_type == "select":
    #             widget = Select(
    #                 id=key,
    #                 label=key,
    #                 initial=initial,
    #                 tooltip=tooltip,
    #                 description=description,
    #                 values=values,
    #                 initial_value=initial,
    #             )
    #         elif widget_type == "slider":
    #             widget = Slider(
    #                 id=key,
    #                 label=key,
    #                 initial=initial,
    #                 tooltip=tooltip,
    #                 description=description,
    #                 min=min,
    #                 max=max,
    #             )
    #         elif widget_type == "numberinput":
    #             widget = NumberInput(
    #                 id=key,
    #                 label=key,
    #                 initial=initial,
    #                 tooltip=tooltip,
    #                 description=description,
    #             )
    #         elif widget_type == "textinput":
    #             widget = TextInput(
    #                 id=key,
    #                 label=key,
    #                 initial=initial,
    #                 tooltip=tooltip,
    #                 description=description,
    #             )
    #         elif widget_type == "switch":
    #             widget = Switch(
    #                 id=key,
    #                 label=key,
    #                 initial=initial,
    #                 tooltip=tooltip,
    #                 description=description,
    #             )
    #         elif widget_type == "tags":
    #             widget = Tags(
    #                 id=key,
    #                 label=key,
    #                 initial=initial or [],
    #                 tooltip=tooltip,
    #                 description=description,
    #                 values=values or [],
    #             )
    #         if widget:
    #             widgets.append(widget)
    #     return widgets



