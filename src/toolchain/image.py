
import json
import re
import asyncio
import dataclasses
# import replicate
import time
import aiofiles

from typing import (
    Callable, List, Dict, Any, Optional, Literal, Tuple, Type, TypeVar, Union, Mapping,
    BinaryIO, TextIO, Awaitable, Coroutine, TYPE_CHECKING,
)
from typing_extensions import TypeAlias

from aiohttp import ClientConnectorError

from pathlib import Path

from pydantic import BaseModel, Field, root_validator, ConfigDict
from pydantic.dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from dataclasses_json.core import (Json, _ExtendedEncoder, _asdict,
                                   _decode_dataclass)

from chainlit import Action, Audio, Image, Video, File, Message, Step, AskUserMessage
from chainlit.element import Element, ElementType, ElementDisplay, ElementSize
from chainlit.input_widget import InputWidget, Select, Slider, NumberInput
from chainlit.types import InputWidgetType

from replicate.prediction import Prediction, Predictions
import replicate

from toolchain.models import DataClassJsonMixinPro



_image_toolchain_image_path_map: Mapping[str, Mapping[Path, 'ImageToolchain']] = dict()
_image_toolchain_image_path_map_by_id: Mapping[str, Mapping[str, Path]] = dict()

CreationResponse = TypeVar('CreationResponse')

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ImageToolchain(DataClassJsonMixinPro):
    from toolchain.types import ImageToolchainTemplates
    unsafe_html: bool = dataclasses.field(default=True, repr=False, kw_only=True)
    destination_path: Path = dataclasses.field(default=Path('public'), kw_only=True)
    timestamp: int = dataclasses.field(default_factory=lambda: int(time.time()), kw_only=True)
    templates_class: Type[ImageToolchainTemplates] = dataclasses.field(default=ImageToolchainTemplates, repr=False, kw_only=True)
    public_id: Optional[str] = dataclasses.field(default=None, kw_only=True)
    source_image: Element = dataclasses.field(default=...)
    message: Optional[Message] = dataclasses.field(default=None)
    creation_responses: List[CreationResponse] = dataclasses.field(default_factory=list)
    creation_id_step_map: Mapping[str, Step] = dataclasses.field(default_factory=dict)
    
    @property
    def templates(self):
        return self.templates_class(unsafe_html=self.unsafe_html)
    @property
    def source_path(self) -> Path:
        return Path(self.source_image.path)
    @property
    def image_file_name(self) -> str:
        value = self.templates.FILE_NAME_TEMPLATE.format(
            PREFIX=self.source_image.name.removesuffix(self.source_path.suffix).replace(',', ''),
            TIMESTAMP=self.timestamp,
            EXT=self.source_path.suffix,
        )
        return value
    @property
    def image_path(self) -> Path:
        self.destination_path.mkdir(parents=True, exist_ok=True)
        image_path = self.destination_path.joinpath(self.image_file_name)
        if not image_path.exists():
            image_path.touch(mode=0o777, exist_ok=True)
        return image_path
    @property
    def message_id(self) -> Optional[str]:
        return self.message.id if self.message else None
    @property
    def image_actions(self) -> List[Action]:
        return self.message.actions if self.message else []
    @property
    def image_action_ids(self) -> List[str]:
        return [action.id for action in self.image_actions]
    @property
    def image_element(self) -> Optional[Image]:
        """The image element representing a potential svd prediction."""
        return self.message.elements[0] if self.message else None
    @property
    def steps(self) -> List[Step]:
        return list(self.creation_id_step_map.values())
    @property
    def step_ids(self) -> List[str]:
        return [step.id for step in self.steps]
    @property
    def creation_ids(self) -> List[str]:
        return list(self.creation_id_step_map.keys())
    
    def save(self, public_id: Optional[str] = None):
        """Save the toolchain to the datastore."""
        global _image_toolchain_image_path_map
        public_id = public_id or self.public_id
        if public_id is None:
            raise ValueError("Must provide a `public_id`.")
        _image_toolchain_image_path_map = {
            **_image_toolchain_image_path_map,
            public_id: {
                **_image_toolchain_image_path_map.get(public_id, {}),
                self.image_path: self,
            },
        }
        return _image_toolchain_image_path_map
    
    def save_ids(self, public_id: str, override_ids: List[str] = []):
        global _image_toolchain_image_path_map_by_id
        print("override_ids or self.creation_ids:\n", override_ids or self.creation_ids)
        _image_toolchain_image_path_map_by_id = {
            **_image_toolchain_image_path_map_by_id,
            public_id: {
                **_image_toolchain_image_path_map_by_id.get(public_id, dict()),
                **{creation_id: self.image_path for creation_id in override_ids or self.creation_ids},
            }
        }
        lookup_test_a = _image_toolchain_image_path_map_by_id.get(public_id, dict())
        print("lookup_test_a: ", lookup_test_a)
        lookup_test_bs = [_image_toolchain_image_path_map_by_id.get(public_id, dict()).get(id, None) for id in override_ids or self.creation_ids]
        print("lookup_test_bs: ", lookup_test_bs)
        return None
    
    @classmethod
    def from_save(cls, public_id: str, image_path: Union[str, Path]):
        """Load the toolchain from the datastore."""
        global _image_toolchain_image_path_map
        print("keys: ", list(_image_toolchain_image_path_map.keys()), end="; ", flush=True)
        print("found public_id in keys: ", public_id in _image_toolchain_image_path_map.keys(), end="; ", flush=True)
        if image_path is None:
            print("image_path is None\n")
            print(f"_image_toolchain_image_path_map.keys(): {_image_toolchain_image_path_map.keys()}\n")
            print(f"_image_toolchain_image_path_map_by_id.keys(): {_image_toolchain_image_path_map_by_id.keys()}\n")
            return None
        image_toolchain = _image_toolchain_image_path_map.get(
            public_id, dict()
        ).get(Path(image_path), None)
        if image_toolchain:
            image_toolchain.public_id = public_id
        return image_toolchain
    
    @classmethod
    def from_saved_id(cls, public_id: str, creation_id: str):
        """Load the toolchain from the datastore."""
        global _image_toolchain_image_path_map_by_id
        image_path = _image_toolchain_image_path_map_by_id.get(
            public_id, dict()
        ).get(creation_id, None)
        return cls.from_save(public_id, image_path)
    
    @classmethod
    def from_saved_action(cls, public_id: str, action: Action):
        """Load the toolchain from the datastore."""
        _, image_path = action.value.split(",")
        return cls.from_save(public_id, Path(image_path))
    
    def get_step(self, creation_id: str) -> Optional[Step]:
        # return next((step for step in self.steps if step.id == step_id), None)
        return self.creation_id_step_map.get(creation_id)
    
    @classmethod
    async def from_image(
        cls,
        public_id: str,
        image_file: Element,
        *,
        destination_path: Path = Path('public'),
        unsafe_html: bool = True,
    ):
        instance = cls(
            image_file,
            public_id=public_id,
            unsafe_html=unsafe_html,
            destination_path=destination_path,
        )
        return await instance.initialize()
    
    async def initialize(self):
        """First step in the toolchain."""
        await self._copy_image()
        self.message = self._create_message()
        await self.message.send()
        self.save()
        return self
    
    async def _copy_image(self):
        """Copy the source image to the `destination_path` / `FILE_NAME_TEMPLATE`."""
        async with aiofiles.open(self.source_path, "rb") as f:
            self.image_path.write_bytes(await f.read())
        return self.image_path
    
    def _create_image_element(self) -> Optional[Image]:
        """Create an image element representing a potential svd prediction."""
        image_element = Image(
            name=self.image_file_name,
            path=self.image_path.as_posix(),
            size="small",
            display="inline",
        )
        return image_element
    
    def _create_actions(self, image_path: Union[str, Path], action_name: str = "Create video") -> List[Action]:
        """Create actions for starting the creation process.
        args:
            image_path: The path to the image to convert.
            action_name: The name of the `chainlit.action_callback` to call when `Action` is clicked.
        """
        batch_sizes = [1, 2, 3, 5]
        s: Callable[[int], Literal['s', '']] = lambda i: 's' if i > 1 else ''
        actions = [
            Action(
                name=action_name,
                value=f"{i},{Path(image_path).as_posix()}",
                label=f"Start {i} creation{s(i)}",
                description=f"Start {i} creation{s(i)} from the image in the message.",
            ) for i in batch_sizes
        ]
        actions.append(
            Action(
                name=action_name,
                value=f"0,{Path(image_path).as_posix()}",
                label=f"Specify how many",
                description="Specify how many creations to start from the image in the message.",
            )
        )
        return actions

    def _create_cancel_actions(self, creation_id: str, action_name: str = "Cancel creation") -> List[Action]:
        cancel_actions = [
            Action(
                name=action_name,
                value=creation_id,
                label="Cancel creation",
                description="Cancel the associated creation process.",
                collapsed=False,
            ),
        ]
        return cancel_actions
    
    def _create_message(self):
        """Create the message for the job.
        Assumes that the image has already been copied to the destination path.
        """
        image_element = self._create_image_element()
        image_actions = self._create_actions(self.image_path)
        message = Message(
            content=self.templates.CREATE_VIDEO_FROM_TEMPLATE.format(
                IMAGE_NAME=image_element.name,
            ),
            actions=image_actions,
            elements=[image_element],
        )
        return message
    
    async def _change_message(self, content: str, delivery: Literal["send", "update"] = "update"):
        if self.message is None:
            return self.message_id
        self.message.content = content
        if delivery == "send":
            return await self.message.send()
        return await self.message.update()
    
    async def _remove_actions(self):
        return await asyncio.gather(*[action.remove() for action in self.image_actions])
    
    async def _create_step(self, creation_id: str, progress: int = 0):        
        from toolchain.utils import progress_bar_simple
        step = Step(
            name=self.message.author,
            parent_id=self.message_id,
            type="tool",
            root=True,
        )
        await step.stream_token(progress_bar_simple(progress))
        await step.update()
        self.creation_id_step_map[creation_id] = step
        return step
    
    @classmethod
    async def load_from_action(
        cls,
        public_id: str,
        action: Action,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ):
        """Pre-second step in the toolchain.
        Call inside the function with the `chainlit.action_callback` decorator
        corresponding to the `Action` created by `self._create_actions`.
        args:
            action: The `Action` that was clicked.
            creation_fn: The function that creates the video. It needs to return creation ids as
                a `List[str]` or a `CreationResponse` object that can be parsed into a `List[str]`.
            parse_response_fn: An optional function if the `creation_fn` requires an additional step
                to parse the response into a `List[str]` of creation ids. Defaults to a pass-through.
        """
        # Extract the image path from the action to load the toolchain
        _, image_path = cls.parse_action(action)
        image_toolchain = cls.from_save(public_id, image_path)
        
        ids = await image_toolchain.handle_action(
            public_id=public_id,
            action=action,
            creation_fn=creation_fn,
            parse_response_fn=parse_response_fn,
        )
        return image_toolchain
    
    @staticmethod
    def parse_action(action: Action) -> Tuple[int, Path]:
        """Parse the `Action` that was clicked."""
        from toolchain.utils import safe_int_parse
        count, file_path = action.value.split(',')
        count = safe_int_parse(count, 0)
        count = min(10, max(0, count))
        return count, Path(file_path)
    
    async def handle_action(
        self,
        public_id: str,
        action: Action,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ):
        """Handle the `Action` that was clicked."""
        from toolchain.utils import safe_int_parse
        
        count, image_path = self.parse_action(action)
        if count == 0:
            ask_message = AskUserMessage(
                content="How many creations would you like to start?",
                timeout=3600,
            )
            response = await ask_message.send()
            if response:
                count = safe_int_parse(response["output"], 0)
                await ask_message.remove()
                await Message(content="", id=response["id"]).remove()
        
        # Ensure at least one creation is started and no more than 10
        count = min(10, max(1, count))
        await self._remove_actions()
        await self._change_message(
            content=self.templates.CREATING_COUNT_VIDEOS_FROM_TEMPLATE.format(
                COUNT=count,
                PLURAL="s" if count > 1 else "",
                IMAGE_NAME=self.image_file_name,
            ),
            delivery="update",
        )
        self.creation_responses = await creation_fn(count, image_path)
        creation_ids = [parse_response_fn(response) for response in self.creation_responses]
        self.save_ids(public_id, creation_ids)
        steps = await asyncio.gather(*[self._create_step(creation_id) for creation_id in creation_ids])
        return self.creation_ids
    
    async def update_progress(self, creation_id: str, progress: int, output: Optional[str] = None):
        """Update the progress displayed for an active creation."""
        from toolchain.utils import progress_bar_simple
        step = self.get_step(creation_id)
        if step is None:
            return print(f"Warning: Could not find step for creation_id: {creation_id}")
        if output is not None:
            old_ext = self.image_path.name.split('.').pop()
            new_ext = output.split('.').pop()
            new_name = self.image_path.name.replace(old_ext, new_ext)
            dl_video_text = self.templates.DOWNLOAD_FILE_MARKDOWN_TEMPLATE.format(
                FILE_NAME=new_name,
                FILE_URL=output,
            )
            step.output = dl_video_text
            video_name = self.image_file_name.split(".").pop(0)
            print("video_name: ", video_name)
            video_url = output
            print("video_url: ", video_url)
            video_display = "inline"
            video_size = "medium"
            print(f"{video_size}-{video_display}")
            video_for_id = step.id
            print("video_for_id: ", video_for_id)
            video: Video = None
            try:
                video = Video(
                    name=video_name,
                    url=video_url,
                    display=video_display,
                    size=video_size,
                    for_id=video_for_id,
                )
            except Exception as e:
                print("Error creating video: ", e)
            if video is None:
                return print("video is None")
            step.elements = [video]
            # step.elements = [
            #     Video(
            #         name=self.image_path.name,
            #         url=output,
            #         # path=output,
            #         display="inline",
            #         size="medium",
            #         for_id=step.id,
            #     ),
            # ]
            try:
                await step.send()
            except Exception as e:
                print("Error sending step: ", e)
        else:
            step.output = progress_bar_simple(progress)
            try:
                await step.update()
            except Exception as e:
                print("Error updating step: ", e)
        return None









