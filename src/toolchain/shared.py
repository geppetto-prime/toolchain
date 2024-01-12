from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Generic, TypeVar, Type, Callable, Mapping

from chainlit.types import InputWidgetType
from pydantic.dataclasses import Field, dataclass

import chainlit.input_widget

T = TypeVar("T")



@dataclass
class InputWidget(chainlit.input_widget.InputWidget, Generic[T]):
    initial: Optional[T] = None
    id: Optional[str] = "None"
    label: Optional[str] = "None"

    @classmethod
    def upgrade(cls, widget: chainlit.input_widget.InputWidget, type: Type[T]) -> 'InputWidget[T]':
        # Unpack all attributes from the original widget
        widget_dict = widget.to_dict()
        return cls(**widget_dict)


@dataclass
class Switch(InputWidget, Generic[T]):
    """Useful to create a switch input."""

    type: InputWidgetType = "switch"
    initial: T = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "tooltip": self.tooltip,
            "description": self.description,
        }


@dataclass
class Slider(InputWidget, Generic[T]):
    """Useful to create a slider input."""

    type: InputWidgetType = "slider"
    initial: Optional[T] = 0
    min: T = 0
    max: T = 10
    step: T = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "tooltip": self.tooltip,
            "description": self.description,
        }


@dataclass
class Select(InputWidget, Generic[T]):
    """Useful to create a select input."""

    type: InputWidgetType = "select"
    initial: Optional[T] = "None"
    initial_index: Optional[int] = None
    initial_value: Optional[T] = None
    values: List[T] = Field(default_factory=lambda: [])
    items: Dict[str, T] = Field(default_factory=lambda: defaultdict(dict))

    def __post_init__(
        self,
    ) -> None:
        super().__post_init__()

        if not self.values and not self.items:
            raise ValueError("Must provide values or items to create a Select")

        if self.values and self.items:
            raise ValueError(
                "You can only provide either values or items to create a Select"
            )

        if not self.values and self.initial_index is not None:
            raise ValueError(
                "Initial_index can only be used in combination with values to create a Select"
            )

        if self.items:
            self.initial = self.initial_value
        elif self.values:
            self.items = {value: value for value in self.values}
            self.initial = (
                self.values[self.initial_index]
                if self.initial_index is not None
                else self.initial_value
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "items": [
                {"label": id, "value": value} for id, value in self.items.items()
            ],
            "tooltip": self.tooltip,
            "description": self.description,
        }


@dataclass
class TextInput(InputWidget, Generic[T]):
    """Useful to create a text input."""

    type: InputWidgetType = "textinput"
    initial: Optional[T] = None
    placeholder: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "placeholder": self.placeholder,
            "tooltip": self.tooltip,
            "description": self.description,
        }

# TF = TypeVar("TF", float, int)

@dataclass
class NumberInput(InputWidget, Generic[T]):
    """Useful to create a number input."""

    type: InputWidgetType = "numberinput"
    initial: Optional[T] = None
    placeholder: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "placeholder": self.placeholder,
            "tooltip": self.tooltip,
            "description": self.description,
        }


@dataclass
class Tags(InputWidget, Generic[T]):
    """Useful to create an input for an array of strings."""

    type: InputWidgetType = "tags"
    initial: List[T] = Field(default_factory=lambda: [])
    values: List[T] = Field(default_factory=lambda: [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "label": self.label,
            "initial": self.initial,
            "tooltip": self.tooltip,
            "description": self.description,
        }


widget_map: Mapping[InputWidgetType, Type[InputWidget[T]]] = {
    "textinput": TextInput,
    "numberinput": NumberInput,
    "select": Select,
    "slider": Slider,
    "switch": Switch,
    "tags": Tags,
}

