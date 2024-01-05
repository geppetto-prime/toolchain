"""Main code for the image toolchain functionality."""

import asyncio
import dataclasses
import time
import aiofiles

from pathlib import Path

from urllib.request import urlretrieve

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
from toolchain.utils import safe_name_parse, safe_int_parse, progress_bar_simple
from toolchain.session import SessionId, SessionToolchain

CreationResponse = TypeVar('CreationResponse')
ImageToolchainType = TypeVar('ImageToolchainType', bound='ImageToolchain')

_image_path_map: MutableMapping[SessionId, MutableMapping[Path, Type[ImageToolchainType]]] = dict()
_image_path_map_by_id: MutableMapping[SessionId, MutableMapping[str, Path]] = dict()


def strip_timestamp(text: str) -> str:
    if text[len(text) - 10:].isnumeric():
        return text[:len(text) - 10].strip()
    return text

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
    session_toolchain: Optional[SessionToolchain] = dataclasses.field(default=None, repr=False, kw_only=True)
    source_image_path: str = dataclasses.field(default=...)
    source_image_name: str = dataclasses.field(default=...)
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
        return Path(self.source_image_path)
    
    @property
    def _image_file_name(self) -> str:
        """Returns an image file name formatted with the `FILE_NAME_TEMPLATE`."""
        prefix = self.source_image_name.removesuffix(self.source_path.suffix)
        value = self.templates.FILE_NAME_TEMPLATE.format(
            PREFIX=safe_name_parse(prefix),
            TIMESTAMP=self.timestamp,
            EXT=self.source_path.suffix.strip('.'),
        )
        return value
    
    @property
    def image_path(self) -> Path:
        """The image path to the initial image (copy).
        Always returns a valid file path, creating directories and a file as needed."""
        self.destination_path.mkdir(parents=True, exist_ok=True)
        image_path = self.destination_path.joinpath(self._image_file_name)
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
    
    @staticmethod
    def image_path_map(session_id: SessionId) -> MutableMapping[Path, 'ImageToolchain']:
        """Returns the global image_path map."""
        global _image_path_map
        if session_id not in _image_path_map:
            _image_path_map.update({session_id: dict()})
        return _image_path_map.get(session_id, dict())
    
    @staticmethod
    def image_path_map_by_id(session_id: SessionId) -> MutableMapping[str, Path]:
        """Returns the global image path by id map."""
        global _image_path_map_by_id
        if session_id not in _image_path_map_by_id:
            _image_path_map_by_id.update({session_id: dict()})
        return _image_path_map_by_id.get(session_id, dict())
    
    def save_to_image_path_map(self, session_id: Optional[SessionId] = None) -> None:
        """Saves the toolchain state to a global map.

        Args:
            session_id (Optional[SessionId]): Private identifier for the user's `Chainlit` context.
        """
        session_id = session_id or self.session_id
        if session_id is None:
            raise ValueError("Must provide a `session_id`.")
        image_path_map = self.image_path_map(session_id)
        image_path_map.update({self.image_path: self})
        return None
    
    @classmethod
    def _load_from_image_path_map(cls, session_id: SessionId, image_path: Union[str, Path]) -> Union[ImageToolchainType, None]:
        """Loads the toolchain from saved state based on a private session ID and image path."""
        image_path_map = cls.image_path_map(session_id)
        if image_path is None:
            keys = list(image_path_map.keys())
            print("image toolchains:\n", list(image_path_map.values()), end="\n\n")
            print("keys: ", keys)
            image_path_map_by_id = cls.image_path_map_by_id(session_id)
            print("image_path_map_by_id:\n", image_path_map_by_id)
            return None
        instance = image_path_map.get(Path(image_path), None)
        if instance is not None:
            instance.session_id = session_id
        return instance
        
    
    def save_ids_to_image_map_by_id(
        self,
        creation_ids: List[str],
        session_id: Optional[SessionId] = None,
    ) -> None:
        """Saves the mapping of creation IDs to image paths.

        Args:
            creation_ids (List[str]): List of creation IDs to save.
            session_id (Optional[SessionId]): Private identifier for the user's `Chainlit` context.
        """
        session_id = session_id or self.session_id
        image_path_map_by_id = self.image_path_map_by_id(session_id)
        image_path_map_by_id.update({creation_id: self.image_path for creation_id in creation_ids})
        return None
    
    @classmethod
    def load_from_image_path_map(
        cls: Type[ImageToolchainType],
        public_id: str,
        image_path: Union[str, Path],
    ) -> Union[ImageToolchainType, None]:
        """Loads the toolchain from saved state based on a public ID and image path."""
        session = SessionToolchain.from_public_id(public_id)
        instance = cls._load_from_image_path_map(session.session_id, image_path)
        if instance is not None:
            instance.public_id = public_id
            instance.session_id = session.session_id
        return instance
    
    @classmethod
    def load_from_image_path_map_by_id(
        cls: Type[ImageToolchainType],
        public_id: str,
        creation_id: str,
    ) -> Union[ImageToolchainType, None]:
        """Loads the toolchain from saved state based on a public ID and creation ID."""
        session = SessionToolchain.from_public_id(public_id)
        if session is None:
            return None
        image_path_map_by_id = cls.image_path_map_by_id(session.session_id)
        image_path = image_path_map_by_id.get(creation_id, None)
        return cls.load_from_image_path_map(public_id, image_path)
    
    @classmethod
    async def load_from_creation_action(
        cls: Type[ImageToolchainType],
        public_id: str,
        action: Action,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ) -> Union[ImageToolchainType, None]:
        """Load the toolchain from the datastore."""
        count, image_path = cls.parse_creation_action(action)
        instance = cls.load_from_image_path_map(
            public_id=public_id,
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
    
    def on_chat_start(self, session_toolchain: SessionToolchain):
        """Simple, consistant way to load session across Toolchain and Link classes."""
        self.session_toolchain = session_toolchain
        return None
    
    @classmethod
    async def from_prompt(
        cls: Type[ImageToolchainType],
        public_id: str,
        prompt: str,
        *,
        destination_path: Path = Path('public'),
        unsafe_html: bool = True,
    ) -> ImageToolchainType:
        """Create a new image toolchain instance from a prompt."""
        session = SessionToolchain.from_public_id(public_id=public_id)
        image_file = f"{safe_name_parse(prompt)}.png"
        instance = cls(
            source_image_path=image_file,
            source_image_name=image_file,
            session_id=session.session_id,
            unsafe_html=unsafe_html,
            destination_path=destination_path,
        )
        initialized_instance = await instance._initialize_prompt(prompt=prompt)
        return initialized_instance
    
    @classmethod
    async def from_image(
        cls: Type[ImageToolchainType],
        public_id: str,
        image_file: Element,
        *,
        destination_path: Path = Path('public'),
        unsafe_html: bool = True,
        image_size: Optional[ElementSize] = None,
        image_display: Optional[ElementDisplay] = None,
        override_id: Optional[str] = None,
    ) -> ImageToolchainType:
        """Create a new image toolchain instance from a `chainlit.Image`."""
        session = SessionToolchain.from_public_id(public_id=public_id)
        if image_file.path:
            source_image_path = Path(image_file.path)
        elif image_file.url:
            source_image_url_path = Path(image_file.url)
            source_image_name = source_image_url_path.with_suffix("").name
            destination_path.mkdir(parents=True, exist_ok=True)
            source_image_path = destination_path / f"{source_image_name}_{str(int(time.time()))[-5]}{source_image_url_path.suffix}"
            urlretrieve(image_file.url, source_image_path)
        instance = cls(
            source_image_path=source_image_path.as_posix(),
            source_image_name=image_file.name,
            session_id=session.session_id,
            unsafe_html=unsafe_html,
            destination_path=destination_path,
        )
        initialized_instance = await instance._initialize(
            image_size=image_size,
            image_display=image_display,
            override_id=override_id,
        )
        return initialized_instance
    
    async def initialize(self):
        """Initialize the toolchain.
        Used if manually creating an instance."""
        return await self._initialize()
    
    async def _initialize(
        self,
        image_size: Optional[ElementSize] = None,
        image_display: Optional[ElementDisplay] = None,
        override_id: Optional[str] = None,
    ):
        """First step in the toolchain."""
        await self._copy_image()
        self.message = self._create_message(
            image_size=image_size,
            image_display=image_display,
            override_id=override_id,
        )
        await self.message.send()
        self.save_to_image_path_map()
        return self
    
    async def _initialize_prompt(self, prompt: str):
        """First step in the toolchain - prompt version."""
        self.message = self._create_message_prompt(prompt)
        await self.message.send()
        self.save_to_image_path_map()
        return self
    
    async def _copy_image(self):
        """Copy the source image to the `destination_path` / `FILE_NAME_TEMPLATE`."""
        async with aiofiles.open(self.source_path, "rb") as f:
            self.image_path.write_bytes(await f.read())
        return self.image_path
    
    def _create_image_element(
        self,
        size: Optional[ElementSize] = None,
        display: Optional[ElementDisplay] = None,
    ) -> Optional[Image]:
        """Create an image element to track creations."""
        image_element = Image(
            name=self.image_path.name,
            path=self.image_path.as_posix(),
            size=size or "small",
            display=display or "inline",
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
                description=f"Start {i} creation{s(i)} from this message.",
            ) for i in batch_sizes
        ]
        actions.append(
            Action(
                name=action_name,
                value=f"0,{Path(image_path).as_posix()}",
                label=f"Specify how many",
                description="Specify how many creations to start from this message.",
            )
        )
        return actions

    def _create_cancel_actions(self, creation_id: str, action_name: str = "Cancel creation") -> List[Action]:
        cancel_actions = [
            Action(
                name=action_name,
                value=creation_id,
                label=action_name,
                description="Cancel the associated creation process.",
                collapsed=False,
            ),
        ]
        return cancel_actions
    
    def _create_message(
        self,
        creation_type: Literal["image", "video"] = "video",
        image_size: Optional[ElementSize] = None,
        image_display: Optional[ElementDisplay] = None,
        override_id: Optional[str] = None,
        override_content: Optional[str] = None,
    ):
        """Create the message for the job.
        Assumes that the image has already been copied to the destination path.
        """
        image_element = self._create_image_element(size=image_size, display=image_display)
        image_actions = self._create_actions(self.image_path)
        message = Message(
            content=override_content or self.templates.CREATE_FROM_TEMPLATE.format(
                TYPE=creation_type,
                FROM=image_element.name,
            ),
            actions=image_actions,
            elements=[image_element],
            id=override_id,
        )
        return message
    
    def _create_message_prompt(self, prompt: str):
        """Create the message for the job.
        Assumes that the image has already been copied to the destination path.
        """
        image_actions = self._create_actions(self.image_path, action_name="Create image")
        message = Message(
            content=self.templates.CREATE_FROM_TEMPLATE.format(
                TYPE="image",
                FROM=prompt,
            ),
            actions=image_actions,
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
    
    async def _create_steps(self, creation_ids: List[str]) -> List[Step]:
        """Create steps for each id in `creation_ids`.
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
        return await asyncio.gather(*[create_step(creation_id) for creation_id in creation_ids])
    
    
    async def _handle_creations(
        self,
        count: int,
        creation_fn: Callable[[int, Union[str, Path]], Awaitable[List[CreationResponse]]] = lambda count, image_path: [],
        parse_response_fn: Callable[[CreationResponse], str] = lambda response: response,
    ):
        """Handle the `Action` that was clicked."""
        creation_type = "image" if self.image_path.stat().st_size == 0 else "video"
        
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
            content=self.templates.CREATING_COUNT_FROM_TEMPLATE.format(
                TYPE=creation_type,
                COUNT=count,
                PLURAL="s" if count > 1 else "",
                IMAGE_NAME=self.image_path.name,
            ),
            delivery="update",
        )
        # If `image_path` is an empty file that's only been 'touch', treat it as a prompt string:
        if creation_type == "image":
            prompt = self.image_path.name.split(".").pop(0).replace("_", " ")
            try:
                prompt = strip_timestamp(prompt)
                self.creation_responses = await creation_fn(count, prompt)
            except Exception as e:
                print(e)
                self.creation_responses = []
        else:
            try:
                self.creation_responses = await creation_fn(count, self.image_path)
            except Exception as e:
                print(e)
                self.creation_responses = []
        try:
            creation_ids = [parse_response_fn(response) for response in self.creation_responses]
            self.save_ids_to_image_map_by_id(creation_ids=creation_ids)
        except Exception as e:
            print(f"Error '{str(e)}' creating steps for creation responses:", self.creation_responses)
            creation_ids = []
        steps = await self._create_steps(creation_ids=creation_ids)
        return steps
    
    async def update_progress(self, creation_id: str, progress: int, output: Optional[Union[str, List[str]]] = None):
        """Update the progress displayed for an active creation."""
        step = self.get_step(creation_id)
        if step is None:
            return print(f"Warning: Could not find step for creation_id: {creation_id}")
        if output is not None:
            output = [output] if isinstance(output, str) else output
            
            # update the step to display 100%
            step.output = progress_bar_simple(100)
            step.elements = []
            try:
                await step.update()
            except Exception as e:
                print("Error updating finished step: ", e)
            
            for _output in output:
                creation_type = "image" if self.image_path.stat().st_size == 0 else "video"
                old_ext = self.image_path.suffix.strip(".")
                new_ext = _output.split('.').pop()
                new_file_path = self.image_path.with_suffix(f".{new_ext}")
                new_file_name = new_file_path.name
                
                dl_created_file_text = self.templates.render_image_container(
                    title=f"{creation_type.capitalize()} generation complete:",
                    image_name=new_file_name,
                    image_url=_output,
                    display_image=creation_type == "image",
                )
                if new_ext in ['mp4']:
                    video: Video = None
                    try:
                        video = Video(
                            name=new_file_name,
                            url=_output,
                            display="inline",
                            size="medium",
                            for_id=step.id,
                        )
                        if video:
                            step.elements.append(video)
                        step.output = dl_created_file_text
                        await step.send()
                    except Exception as e:
                        print("Error creating video: ", e)
                else:
                    image: Image = None
                    try:
                        temp_image_name = new_file_path.with_suffix("").name
                        temp_image_name = strip_timestamp(temp_image_name).strip("_")
                        temp_image_name = f"{temp_image_name}_{int(time.time())}.{new_ext}"
                        new_file_path = new_file_path.with_name(temp_image_name)
                        urlretrieve(_output, new_file_path)
                        await step.send()
                        
                        temp_image = Image(
                            name=new_file_path.name,
                            path=new_file_path.as_posix(),
                            display="inline",
                            size="medium",
                            for_id=step.id,
                        )
                        image_toolchain = await ImageToolchain.from_image(
                            public_id=self.public_id,
                            image_file=temp_image,
                            destination_path=self.destination_path,
                            image_size="medium",
                            image_display="inline",
                            # override_id=step.id,
                            # override_id=self.message_id,
                        )
                    except Exception as e:
                        print("Error creating image: ", e)
                    return None
        else:
            step.output = progress_bar_simple(progress)
            try:
                await step.update()
            except Exception as e:
                print("Error updating step: ", e)
        return None









