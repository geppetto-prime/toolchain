
import json
import re
import asyncio
import dataclasses
import time
import aiofiles

from typing import (
    Callable, List, Dict, Any, Optional, Literal, Tuple, Type, TypeVar, Union, Mapping,
    BinaryIO, TextIO, Awaitable,
)
from typing_extensions import TypeAlias

from pathlib import Path

from pydantic import BaseModel, Field, root_validator, ConfigDict
from pydantic.dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from dataclasses_json.core import (Json, _ExtendedEncoder, _asdict,
                                   _decode_dataclass)

from chainlit import Action, Audio, Image, Video, File, Message, Step
from chainlit.element import Element, ElementType, ElementDisplay, ElementSize
from chainlit.input_widget import InputWidget, Select, Slider, NumberInput
from chainlit.types import InputWidgetType



from toolchain.models import DataClassJsonMixinPro


# pyautogen[retrievechat] added:
Extensions: TypeAlias = Literal['txt', 'json', 'csv', 'tsv', 'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml', 'pdf']

# unstructured[all-docs] added:
ExtensionsExtended: TypeAlias = Literal['docx', 'doc', 'odt', 'pptx', 'ppt', 'xlsx', 'eml', 'msg', 'epub']


ReplicatePredictionStatus: TypeAlias = Literal["starting", "processing", "succeeded", "failed", "canceled"]
ReplicatePredictionWebhookEventsFilter: TypeAlias = Literal["start", "output", "logs", "completed"]

StableDiffusionVideo_VideoLength: TypeAlias = Literal["14_frames_with_svd", "25_frames_with_svd_xt"]
StableDiffusionVideo_SizingStrategy: TypeAlias = Literal["maintain_aspect_ratio", "crop_to_16_9", "use_image_dimensions"]
SettingsKeys: TypeAlias = Literal["video_length", "sizing_strategy", "frames_per_second", "motion_bucket_id", "cond_aug", "decoding_t", "seed"]


STABLE_DIFFUSION_VIDEO: str = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"



@dataclass
class ParsedReplicatePredictionLog(DataClassJsonMixinPro):
    id: Optional[str] = dataclasses.field(default=None)
    status: Optional[ReplicatePredictionStatus] = dataclasses.field(default=None)
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
    current_step: Optional[str] = dataclasses.field(default=None)
    total_steps_again: Optional[str] = dataclasses.field(default=None)
    

@dataclass
class ReplicatePredictionNotification(DataClassJsonMixinPro):
    id: Optional[str] = dataclasses.field(default=None)
    status: Optional[ReplicatePredictionStatus] = dataclasses.field(default=None)
    percent_complete: Optional[int] = dataclasses.field(default=0)
    output: Optional[str] = dataclasses.field(default=None)


@dataclass
class ImageToolchainTemplates:
    unsafe_html: bool = dataclasses.field(default=True)
    _FILE_NAME_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="{PREFIX}_{TIMESTAMP}{EXT}", init=False, kw_only=True)
    _FILE_NAME_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="{PREFIX}_{TIMESTAMP}{EXT}", init=False, kw_only=True)
    _CREATE_VIDEO_FROM_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="**Create video from:**\n{IMAGE_NAME}", init=False, kw_only=True)
    _CREATE_VIDEO_FROM_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="**Create video from:**\n{IMAGE_NAME}", init=False, kw_only=True)
    _CREATING_COUNT_VIDEOS_FROM_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="**Creating <span class='cl-red-text'>{COUNT}</span> video{PLURAL} from:**\n{IMAGE_NAME}", init=False, kw_only=True)
    _CREATING_COUNT_VIDEOS_FROM_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="**Creating {COUNT} video{PLURAL} from:**\n{IMAGE_NAME}", init=False, kw_only=True)
    _DOWNLOAD_FILE_MARKDOWN_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="""**Video generation complete.**\nDownload [{FILE_NAME}]({FILE_URL})""", init=False, kw_only=True)
    _DOWNLOAD_FILE_MARKDOWN_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="""**Video generation complete.**\nDownload [{FILE_NAME}]({FILE_URL})""", init=False, kw_only=True)
    @property
    def FILE_NAME_TEMPLATE(self) -> str:
        return self._FILE_NAME_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._FILE_NAME_TEMPLATE_SAFE_HTML
    @property
    def CREATE_VIDEO_FROM_TEMPLATE(self) -> str:
        return self._CREATE_VIDEO_FROM_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._CREATE_VIDEO_FROM_TEMPLATE_SAFE_HTML
    @property
    def CREATING_COUNT_VIDEOS_FROM_TEMPLATE(self) -> str:
        return self._CREATING_COUNT_VIDEOS_FROM_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._CREATING_COUNT_VIDEOS_FROM_TEMPLATE_SAFE_HTML
    @property
    def DOWNLOAD_FILE_MARKDOWN_TEMPLATE(self) -> str:
        return self._DOWNLOAD_FILE_MARKDOWN_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._DOWNLOAD_FILE_MARKDOWN_TEMPLATE_SAFE_HTML




@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StableDiffusionVideoParams(DataClassJsonMixinPro):
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



@dataclass
class StableDiffusionVideoResponseUrls:
    get: Optional[str] = Field("")
    cancel: Optional[str] = Field("")


@dataclass
class StableDiffusionVideoResponse:
    id: str = Field("")
    replicate_model: str = Field(STABLE_DIFFUSION_VIDEO)
    input: StableDiffusionVideoParams = Field(StableDiffusionVideoParams())
    # logs: List[str] = Field(default_factory=list)
    logs: str = Field("")
    error: Optional[Any] = Field(None)
    status: Literal["starting", "processing", "succeeded", "failed", "canceled"] = Field("starting")
    created_at: str = Field("")
    started_at: str = Field("")
    webhook: str = Field("")
    webhook_events_filter: List[Literal["start", "output", "logs", "completed"]] = Field(["start", "output", "logs", "completed"])
    urls: StableDiffusionVideoResponseUrls = Field(StableDiffusionVideoResponseUrls())
    
    @property
    def model(self) -> str:
        return self.replicate_model.split(":").pop(0)
    
    @property
    def version(self) -> str:
        return self.replicate_model.split(":").pop()







