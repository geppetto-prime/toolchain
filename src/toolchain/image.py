"""Main code for the image toolchain functionality."""

import asyncio
import dataclasses
import time
import aiofiles

from pathlib import Path
from typing import (
    Callable, List, Dict, Any, Optional, Literal, Tuple, Type, TypeVar, Union, Mapping,
    Awaitable, Coroutine, MutableMapping, MappingView,
)

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from chainlit import Action, Audio, Image, Video, File, Message, Step, AskUserMessage
from chainlit.element import Element, ElementType, ElementDisplay, ElementSize
from chainlit.input_widget import InputWidget, Select, Slider, NumberInput
from chainlit.types import InputWidgetType

from toolchain.models import DataClassJsonMixinPro
from toolchain.types import ImageToolchainTemplates
from toolchain.utils import safe_int_parse, progress_bar_simple
from toolchain.session import SessionId

CreationResponse = TypeVar('CreationResponse')
ImageToolchainType = TypeVar('ImageToolchainType', bound='ImageToolchain')

_image_path_map: MutableMapping[SessionId, MutableMapping[Path, Type[ImageToolchainType]]] = dict()
_image_path_map_by_id: MutableMapping[SessionId, MutableMapping[str, Path]] = dict()


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ImageToolchain(DataClassJsonMixinPro):
    """Class representing an image toolchain for processing and managing images.

    Attributes:
        unsafe_html (bool): Determines if unsafe HTML is allowed in `.chainlit/config.toml`.
        destination_path (Path): Path to save processed images. Default: `public`.
        timestamp (int): Timestamp for image processing.
        templates_class (Type[ImageToolchainTemplates]): Class for managing HTML templates.
        public_id (Optional[str]): Public identifier for the user's `Chainlit` context.
        source_image (Element): Source image element.
        message (Optional[Message]): Associated message.
        creation_responses (List[CreationResponse]): Responses from creation processes.
        creation_id_step_map (Dict[str, Step]): Mapping of creation IDs to steps.
    """
    # Class fields with detailed type annotations and default values
    unsafe_html: bool = dataclasses.field(default=True, repr=False, kw_only=True)
    destination_path: Path = dataclasses.field(default=Path('public'), kw_only=True)
    timestamp: int = dataclasses.field(default_factory=lambda: int(time.time()), kw_only=True)
    templates_class: Type[ImageToolchainTemplates] = dataclasses.field(default=ImageToolchainTemplates, repr=False, kw_only=True)
    public_id: Optional[str] = dataclasses.field(default=None, kw_only=True)
    session_id: Optional[SessionId] = dataclasses.field(default=None, repr=False, kw_only=True)
    source_image: Element = dataclasses.field(default=...)
    message: Optional[Message] = dataclasses.field(default=None)
    creation_responses: List[CreationResponse] = dataclasses.field(default_factory=list)
    creation_id_step_map: Dict[str, Step] = dataclasses.field(default_factory=dict)
    
    @property
    def templates(self) -> ImageToolchainTemplates:
        """Returns the template instance respecting the `unsafe_html` flag."""
        return self.templates_class(unsafe_html=self.unsafe_html)
    
    @property
    def source_path(self) -> Path:
        """Returns the path of the initial image."""
        return Path(self.source_image.path)
    
    @property
    def image_file_name(self) -> str:
        """Returns an image file name formatted with the `FILE_NAME_TEMPLATE`."""
        value = self.templates.FILE_NAME_TEMPLATE.format(
            PREFIX=self.source_image.name.removesuffix(self.source_path.suffix).replace(',', ''),
            TIMESTAMP=self.timestamp,
            EXT=self.source_path.suffix,
        )
        return value
    
    @property
    def image_path(self) -> Path:
        """The image path to the initial image (copy).
        Always returns a valid file path, creating directories and a file as needed."""
        self.destination_path.mkdir(parents=True, exist_ok=True)
        image_path = self.destination_path.joinpath(self.image_file_name)
        if not image_path.exists():
            image_path.touch(mode=0o777, exist_ok=True)
        return image_path
    
    @property
    def message_id(self) -> Optional[str]:
        """Returns the ID of the associated message, once created."""
        return self.message.id if self.message else None
    
    @property
    def image_actions(self) -> List[Action]:
        """Returns a list of creation related actions associated with the image."""
        return self.message.actions if self.message else []
    
    @property
    def image_action_ids(self) -> List[str]:
        """Returns a list of IDs for the creation related actions associated with the image."""
        return [action.id for action in self.image_actions]
    
    @property
    def image_element(self) -> Optional[Image]:
        """Returns the image element acting as a preview for the creation related actions."""
        return self.message.elements[0] if self.message else None
    
    @property
    def steps(self) -> List[Step]:
        """Returns a list of steps associated with the image toolchain creations."""
        return list(self.creation_id_step_map.values())
    
    @property
    def step_ids(self) -> List[str]:
        """Returns a list of steps IDs associated with the image toolchain creations."""
        return [step.id for step in self.steps]
    
    @property
    def creation_ids(self) -> List[str]:
        """Returns a list of creation IDs associated with the image toolchain."""
        image_path_map_by_id = self.image_path_map_by_id(self.public_id)
        return list(image_path_map_by_id.keys())
    @creation_ids.setter
    def creation_ids(self, value: List[str]):
        """Sets the creation IDs associated with the image toolchain.
        Note: Existing creation IDs will NOT be overwritten."""
        image_path_map_by_id = self.image_path_map_by_id(self.public_id)
        image_path_map_by_id.update({creation_id: self.image_path for creation_id in value})
        return None
    
    def image_path_map(self, session_id: Optional[SessionId] = None) -> MutableMapping[Path, 'ImageToolchain']:
        """Returns the global image_path map."""
        global _image_path_map
        session_id = session_id or self.session_id
        if session_id not in _image_path_map:
            _image_path_map.update({session_id: dict()})
        return _image_path_map.get(session_id, dict())
    
    def image_path_map_by_id(self, session_id: Optional[SessionId] = None) -> MutableMapping[str, Path]:
        """Returns the global image path by id map."""
        global _image_path_map_by_id
        session_id = session_id or self.session_id
        if session_id not in _image_path_map_by_id:
            _image_path_map_by_id.update({session_id: dict()})
        return _image_path_map_by_id.get(session_id, dict())
    
    def save(
        self,
        session_id: Optional[SessionId] = None,
    ) -> MutableMapping[Path, ImageToolchainType]:
        """Saves the toolchain state to a global map.

        Args:
            session_id (Optional[SessionId]): Private identifier for the user's `Chainlit` context.

        Raises:
            ValueError: If session_id is not provided or not set in the class field.

        Returns:
            Mapping[str, Mapping[Path, 'ImageToolchain']]: Updated global toolchain map.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise ValueError("Must provide a `session_id`.")
        image_path_map = self.image_path_map(session_id)
        image_path_map.update({self.image_path: self})
        return image_path_map
    
    def save_ids(
        self,
        session_id: Optional[SessionId] = None,
        override_ids: Optional[List[str]] = None,
    ) -> None:
        """Saves the mapping of creation IDs to image paths.

        Args:
            session_id (Optional[SessionId]): Private identifier for the user's `Chainlit` context.
            override_ids (Optional[List[str]]): List of creation IDs to override.
        """
        session_id = session_id or self.session_id
        ids_to_save = override_ids or self.creation_ids
        image_path_map_by_id = self.image_path_map_by_id(session_id)
        image_path_map_by_id.update({creation_id: self.image_path for creation_id in ids_to_save})
        return None
    
    @classmethod
    def load_from_image_path(
        cls: Type[ImageToolchainType],
        session_id: SessionId,
        image_path: Union[str, Path],
    ) -> Union[ImageToolchainType, None]:
        """Loads the toolchain from saved state based on a public ID and image path.

        Args:
            session_id (SessionId): Private identifier for the user's `Chainlit` context.
            image_path (Union[str, Path]): The path of the image associated with the toolchain.

        Returns:
            ImageToolchain: An instance of ImageToolchain if found, else None.
        """
        global _image_path_map
        if session_id not in _image_path_map:
            print(f"session_id: {session_id} not in _image_path_map: {_image_path_map}")
            return None
        
        image_path_map = _image_path_map.get(session_id, dict())
        
        if image_path is None:
            print("image_path is None\n")
            print(f"image_path_map.keys(): {list(image_path_map.keys())}\n")
            return None
        
        image_toolchain = image_path_map.get(Path(image_path), None)
        if image_toolchain:
            image_toolchain.session_id = session_id
        return image_toolchain
    
    @classmethod
    def load_from_creation_id(
        cls: Type[ImageToolchainType],
        session_id: SessionId,
        creation_id: str,
    ) -> Union[ImageToolchainType, None]:
        """Loads the toolchain from saved state based on a public ID and creation ID.

        Args:
            session_id: (SessionId): Private identifier for the user's `Chainlit` context.
            creation_id (str): The creation ID associated with the toolchain.

        Returns:
            ImageToolchain: An instance of ImageToolchain if found, else None.
        """
        global _image_path_map_by_id
        if session_id not in _image_path_map_by_id:
            print(f"session_id: {session_id} not in _image_path_map_by_id: {_image_path_map_by_id}")
            return None
        
        image_path_map_by_id = _image_path_map_by_id.get(session_id, dict())
        image_path = image_path_map_by_id.get(creation_id, None)
        
        return cls.load_from_image_path(session_id=session_id, image_path=image_path)
    
    @classmethod
    async def load_from_creation_action(
        cls: Type[ImageToolchainType],
        session_id: str,
        action: Action,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ) -> Union[ImageToolchainType, None]:
        """Load the toolchain from the datastore."""
        count, image_path = cls.parse_creation_action(action)
        instance = cls.load_from_image_path(
            session_id=session_id,
            image_path=image_path,
        )
        if instance is None:
            return None
        _ = await instance._handle_creations(
            count=count,
            creation_fn=creation_fn,
            parse_response_fn=parse_response_fn,
        )
        return instance
    
    @staticmethod
    def parse_creation_action(action: Action) -> Tuple[int, Path]:
        """Parse the value of the creation action."""
        count, image_path = action.value.split(",")
        count = safe_int_parse(count, 0)
        return count, Path(image_path)
    
    def get_step(self, creation_id: str) -> Optional[Step]:
        # return next((step for step in self.steps if step.id == step_id), None)
        return self.creation_id_step_map.get(creation_id, None)
    
    @classmethod
    async def from_image(
        cls: Type[ImageToolchainType],
        session_id: SessionId,
        image_file: Element,
        *,
        destination_path: Path = Path('public'),
        unsafe_html: bool = True,
    ) -> Union[ImageToolchainType, None]:
        """Create a new image toolchain instance from a `chainlit.Image`."""
        instance = cls(
            image_file,
            session_id=session_id,
            unsafe_html=unsafe_html,
            destination_path=destination_path,
        )
        return await instance._initialize()
    
    async def initialize(self):
        """Initialize the toolchain.
        Used if manually creating an instance."""
        return await self._initialize()
    
    async def _initialize(self):
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
        """Create an image element to track creations."""
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
    
    async def _remove_creation_actions(self):
        return await asyncio.gather(*[action.remove() for action in self.image_actions])
    
    async def _create_steps(self) -> List[Step]:
        """Create steps for each `creation_id` in `self.creation_ids`.
        Returns a list of `Step` objects."""
        async def create_step(creation_id: str):        
            progress: int = 0
            step = Step(
                name=self.message.author,
                parent_id=self.message_id,
                type="tool",
                root=True,
            )
            await step.stream_token(progress_bar_simple(progress))
            await step.update()
            self.creation_id_step_map.update({creation_id: step})
            return step
        return await asyncio.gather(*[create_step(creation_id) for creation_id in self.creation_ids])
    
    
    async def _handle_creations(
        self,
        count: int,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ):
        """Handle the `Action` that was clicked."""
        
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
        await self._remove_creation_actions()
        await self._change_message(
            content=self.templates.CREATING_COUNT_VIDEOS_FROM_TEMPLATE.format(
                COUNT=count,
                PLURAL="s" if count > 1 else "",
                IMAGE_NAME=self.image_file_name,
            ),
            delivery="update",
        )
        self.creation_responses = await creation_fn(count, self.image_path)
        self.creation_ids = [parse_response_fn(response) for response in self.creation_responses]
        steps = await self._create_steps()
        return steps
    
    async def update_progress(self, creation_id: str, progress: int, output: Optional[str] = None):
        """Update the progress displayed for an active creation."""
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









