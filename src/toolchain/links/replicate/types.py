
import dataclasses

from typing import List, Optional, Literal
from typing_extensions import TypeAlias


from pydantic import BaseModel, Field, root_validator, ConfigDict
from pydantic.dataclasses import dataclass

from chainlit.types import InputWidgetType


STABLE_VIDEO_DIFFUSION_MODEL: str = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
PLAYGROUND_V2_1024PX_AESTHETIC: str = "playgroundai/playground-v2-1024px-aesthetic:42fe626e41cc811eaf02c94b892774839268ce1994ea778eba97103fe1ef51b8"

PredictionStatus: TypeAlias = Literal["starting", "processing", "succeeded", "failed", "canceled"]
PredictionWebhookEventsFilter: TypeAlias = Literal["start", "output", "logs", "completed"]


StableDiffusionVideo_VideoLength: TypeAlias = Literal["14_frames_with_svd", "25_frames_with_svd_xt"]
StableDiffusionVideo_SizingStrategy: TypeAlias = Literal["maintain_aspect_ratio", "crop_to_16_9", "use_image_dimensions"]
StableDiffusionVideo_SettingsKeys: TypeAlias = Literal["video_length", "sizing_strategy", "frames_per_second", "motion_bucket_id", "cond_aug", "decoding_t", "seed"]


PlaygroundV2_1024pxAesthetic_Scheduler: TypeAlias = Literal[
    "DDIM", "DPMSolverMultistep", "HeunDiscrete", "KarrasDPM", "K_EULER_ANCESTRAL", "K_EULER", "PNDM", "DPM++_SDE_Karras"
]


@dataclass
class StableDiffusionVideoParamsMetadata:
    description: str = dataclasses.field(default="")
    tooltip: str = dataclasses.field(default="")
    values: List[str] = dataclasses.field(default_factory=list)
    type: InputWidgetType = dataclasses.field(default="text")
    ge: Optional[int] = dataclasses.field(default=None)
    le: Optional[int] = dataclasses.field(default=None)




