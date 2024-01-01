"""Extend dataclasses_json to support merging and other useful features."""
import dataclasses
import json

from typing import Dict, List, Union, TypeVar, Type, Callable, Optional, Any, cast
from typing_extensions import TypeAlias

from pydantic import BaseModel, Field, ConfigDict
from pydantic.dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin
from dataclasses_json.core import Json, _asdict, _decode_dataclass

A = TypeVar('A', bound="DataClassJsonMixinPro")


@dataclass
class DataClassJsonMixinPro(DataClassJsonMixin):
    """A mixin class that provides additional functionality for data classes with JSON serialization."""

    def to_dict(self, encode_json=False, infer_missing=True) -> Dict[str, Json]:
        """Converts the data class instance to a dictionary.

        Args:
            encode_json (bool, optional): Whether to encode JSON values. Defaults to False.
            infer_missing (bool, optional): Whether to infer missing values. Defaults to True.

        Returns:
            Dict[str, Json]: The dictionary representation of the data class instance.
        """
        _data = _asdict(self, encode_json=encode_json)
        if infer_missing:
            return _data
        return {**{k: v for k, v in _data.items() if v is not None}}

    @classmethod
    def from_dict(cls: Type[A], kvs: Json, *, infer_missing=False) -> A:
        """Creates a data class instance from a dictionary.

        Args:
            kvs (Json): The dictionary containing the data.
            infer_missing (bool, optional): Whether to infer missing values. Defaults to False.

        Returns:
            A: The data class instance.
        """
        return _decode_dataclass(cls, kvs, infer_missing)

    @classmethod
    def from_merge(cls: Type[A], *others: Union[Type[A], Dict[str, Json]], encode_json=False) -> A:
        """Creates a data class instance by merging multiple instances or dictionaries.

        Args:
            *others (Union[Type[A], Dict[str, Json]]): The instances or dictionaries to merge.
            encode_json (bool, optional): Whether to encode JSON values. Defaults to False.

        Returns:
            A: The merged data class instance.
        """
        to_merge = list(others)
        if len(to_merge) == 0:
            return cls()
        merged_data = cls.from_dict(to_merge.pop(0))
        for other in to_merge:
            merged_data = merged_data.merge(other, encode_json=encode_json)
        return merged_data

    def merge(self, *others: Union[Type[A], Dict[str, Json]], encode_json=False) -> A:
        """Merges the data class instance with other instances or dictionaries.

        Args:
            *others (Union[Type[A], Dict[str, Json]]): The instances or dictionaries to merge.
            encode_json (bool, optional): Whether to encode JSON values. Defaults to False.

        Returns:
            A: The merged data class instance.
        """
        for other in others:
            other_dict = other.to_dict(encode_json=encode_json, infer_missing=False) if isinstance(other, DataClassJsonMixinPro) else other
            for key, value in other_dict.items():
                if value is not None:
                    setattr(self, key, value)
        return self

    def to_merged_dict(self, other: Union[Type[A], Dict[str, Json]], encode_json=False, infer_missing=True) -> Dict[str, Json]:
        """Merges the data class instance with another instance or dictionary and converts it to a dictionary.

        Args:
            other (Union[Type[A], Dict[str, Json]]): The instance or dictionary to merge.
            encode_json (bool, optional): Whether to encode JSON values. Defaults to False.
            infer_missing (bool, optional): Whether to infer missing values. Defaults to True.

        Returns:
            Dict[str, Json]: The merged dictionary representation of the data class instance.
        """
        return self.merge(other, encode_json=encode_json).to_dict(encode_json=encode_json, infer_missing=infer_missing)

    def __add__(self, other: Union[Type[A], Dict[str, Json], 'DataClassJsonMixinPro']) -> 'DataClassJsonMixinPro':
        """Adds another instance or dictionary to the data class instance.

        Args:
            other (Union[Type[A], Dict[str, Json], 'DataClassJsonMixinPro']): The instance or dictionary to add.

        Returns:
            DataClassJsonMixinPro: The merged data class instance.
        
        Raises:
            TypeError: If the operand type is not supported.
        """
        if not isinstance(other, DataClassJsonMixinPro) and not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        return self.merge(other)

    @classmethod
    def field_lookup(cls: Type[A], field_name: str):
        """Looks up the field in the data class by name.

        Args:
            field_name (str): The name of the field.

        Returns:
            Any: The field object.

        """
        if field_name in cls.__dataclass_fields__:
            return cls.__dataclass_fields__[field_name]
        return dataclasses.field(default=None)

    @classmethod
    def field_metadata(cls: Type[A], field_name: str):
        """Gets the metadata of the field in the data class by name.

        Args:
            field_name (str): The name of the field.

        Returns:
            dict: The metadata of the field.

        """
        if field_name in cls.__dataclass_fields__:
            return dict(cls.field_lookup(field_name).metadata)
        return dict()

    @classmethod
    def fields_metadata(cls: Type[A], field_names: Union[str, List[str]] = None):
        """Gets the metadata of the fields in the data class.

        Args:
            field_names (Union[str, List[str]], optional): The names of the fields to get metadata for.
                If None, returns metadata for all fields. Defaults to None.

        Returns:
            dict: The metadata of the fields.

        """
        field_names = [field_names] if isinstance(field_names, str) else field_names
        metadata = {
            key: dict(field.metadata)
            for key, field in cls.__dataclass_fields__.items()
            if field_names is None or key in field_names
        }
        return metadata







