

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
    def to_dict(self, encode_json=False, infer_missing=True) -> Dict[str, Json]:
        _data = _asdict(self, encode_json=encode_json)
        if infer_missing:
            return _data
        return {**{k: v for k, v in _data.items() if v is not None}}
    @classmethod
    def from_dict(cls: Type[A],
                  kvs: Json,
                  *,
                  infer_missing=False) -> A:
        return _decode_dataclass(cls, kvs, infer_missing)
    @classmethod
    def from_merge(cls: Type[A], *others: Union[Type[A], Dict[str, Json]], encode_json=False) -> A:
        to_merge = list(others)
        if len(to_merge) == 0:
            return cls()
        merged_data = cls.from_dict(to_merge.pop(0))
        for other in to_merge:
            merged_data = merged_data.merge(other, encode_json=encode_json)
        return merged_data
    def merge(self, *others: Union[Type[A], Dict[str, Json]], encode_json=False) -> A:
        for other in others:
            other_dict = other.to_dict(encode_json=encode_json, infer_missing=False) if isinstance(other, DataClassJsonMixinPro) else other
            for key, value in other_dict.items():
                if value is not None:
                    setattr(self, key, value)
        return self
    def to_merged_dict(self, other: Union[Type[A], Dict[str, Json]], encode_json=False, infer_missing=True) -> Dict[str, Json]:
        return self.merge(other, encode_json=encode_json).to_dict(encode_json=encode_json, infer_missing=infer_missing)
    def __add__(self, other: Union[Type[A], Dict[str, Json], 'DataClassJsonMixinPro']) -> 'DataClassJsonMixinPro':
        if not isinstance(other, DataClassJsonMixinPro) and not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        return self.merge(other)
    @classmethod
    def field_lookup(cls, field_name: str):
        if field_name in cls.__dataclass_fields__:
            return cls.__dataclass_fields__[field_name]
        return dataclasses.field(default=None)
    @classmethod
    def field_metadata(cls, field_name: str):
        if field_name in cls.__dataclass_fields__:
            return dict(cls.field_lookup(field_name).metadata)
        return dict()
    @classmethod
    def fields_metadata(cls, field_names: Union[str, List[str]] = None):
        field_names = [field_names] if isinstance(field_names, str) else field_names
        metadata = {
            key: dict(field.metadata)
            for key, field in cls.__dataclass_fields__.items()
            if field_names is None or key in field_names
        }
        return metadata







