
import asyncio
import functools

from pathlib import Path

from typing import List, MutableMapping, Mapping, Dict, Any, Literal, Union, Optional, TypeVar, Type, Callable, Generic, Coroutine

import chainlit
from chainlit.user_session import UserSession
from chainlit.chat_settings import ChatSettings

from toolchain.shared import InputWidget, Select, Slider, NumberInput, Switch, Tags, TextInput, InputWidgetType, widget_map


T = TypeVar("T")


class SettingsToolchain(Generic[T]):

    class SettingsWidget(Generic[T]):
        widget: InputWidget = None
        user_session: Optional[UserSession] = None
        callback: Optional[Callable[[T, T], Coroutine[Any, Any, None]]] = None
        
        @property
        def key(self) -> str:
            return self.widget.id
        def apply_user_session(self, user_session: UserSession):
            self.user_session = user_session or self.user_session
            return
        def get(self, *args, **kwargs) -> T:
            user_sessions = [arg for arg in [*args, *([kwargs.pop("user_session", None)] or []), *([self.user_session] or [])] if isinstance(arg, UserSession)]
            while len(user_sessions) > 0:
                return user_sessions.pop().get(self.key, self.widget.initial)
        def set(self, value: Optional[T] = None, *args, **kwargs):
            user_sessions = [arg for arg in [*args, *([kwargs.pop("user_session", None)] or []), *([self.user_session] or [])] if isinstance(arg, UserSession)]
            while len(user_sessions) > 0:
                return user_sessions.pop().set(self.key, value)
        
        def __init__(self, widget: InputWidget = None) -> None:
            self.widget = widget
            return
    
    _settings_registry: MutableMapping[str, SettingsWidget] = {}
    _callbacks_registry: MutableMapping[str, Callable[[T, T], Coroutine[Any, Any, None]]] = {}
    
    @classmethod
    def settings_registry(cls) -> MutableMapping[str, SettingsWidget]:
        """
        Convenience method to quickly access `SettingsToolchain._settings_registry`.
        Contains a mapping of settings widgets to be displayed in the chat UI.

        Returns:
            MutableMapping[str, SettingsWidget]: The settings registry.
        """
        return cls._settings_registry
    
    @classmethod
    def callbacks_registry(cls) -> MutableMapping[str, Callable[[T, T], Coroutine[Any, Any, None]]]:
        """
        Convenience method to quickly access `SettingsToolchain._callbacks_registry`.
        Contains a mapping of callbacks to be called when a setting is updated.

        Returns:
            MutableMapping[str, Callable[[T, T], Coroutine[Any, Any, None]]]: The callbacks registry.
        """
        return cls._callbacks_registry
    
    @classmethod
    def register(cls, widget: InputWidget):
        """
        Register a widget in the settings registry. This is called automatically when a function
        is decorated with `@SettingsToolchain.settings`.

        Args:
            widget (InputWidget): The widget to be registered.

        Returns:
            None
        """
        _settings_registry = cls.settings_registry()
        _settings_widget: SettingsToolchain.SettingsWidget = SettingsToolchain.SettingsWidget(widget=widget)
        _settings_registry.update({_settings_widget.key: _settings_widget})
        return
    
    @classmethod
    def register_callback(cls, key: str, callback: Callable[[T, T], Coroutine[Any, Any, None]]) -> None:
        """
        Register a callback in the callbacks registry. This is called automatically when a function,
        decorated with `@SettingsToolchain.settings`, specifies a callback attribute.

        Args:
            key (str): The key of the settings widget to register the callback for.
            callback (Callable[[T, T], Coroutine[Any, Any, None]]): The callback to be registered.

        Returns:
            None
        """
        _callbacks_registry = cls.callbacks_registry()
        _callbacks_registry.update({key: callback})
        return
    
    @classmethod
    def retrieve(cls, key: str) -> Optional['SettingsToolchain.SettingsWidget[T]']:
        """
        Retrieve the settings widget associated with the given key. Functions decorated with
        `@SettingsToolchain.settings` will be available here, their function name will be the key.

        Args:
            key (str): The key of the settings widget to retrieve.

        Returns:
            Optional['SettingsToolchain.SettingsWidget[T]']: The settings widget associated with the key, or None if not found.
        """
        _settings_registry = cls.settings_registry()
        _settings_widget = _settings_registry.get(key, None)
        return _settings_widget
    
    @classmethod
    def retrieve_callback(cls, key: str) -> Optional[Callable[[T, T], Coroutine[Any, Any, None]]]:
        """
        Retrieve the callback associated with the given key. Functions decorated with
        `@SettingsToolchain.settings` will be available here, their function name will be the key.

        Args:
            key (str): The key of the callback to retrieve.

        Returns:
            Optional[Callable[[T, T], Coroutine[Any, Any, None]]]: The callback associated with the key, or None if not found.
        """
        _callbacks_registry = cls.callbacks_registry()
        _callback = _callbacks_registry.get(key, None)
        return _callback
    
    @classmethod
    def inputs(cls) -> List[InputWidget]:
        """
        Returns a list of all registered input widgets which are passed to `cl.ChatSettings(inputs=inputs).send()`
        to display the settings in the chat UI.
        """
        _settings_registry = cls.settings_registry()
        _inputs = [widget.widget for widget in _settings_registry.values()]
        return _inputs
    
    @classmethod
    def settings_list(cls) -> List[SettingsWidget[T]]:
        _settings_registry = cls.settings_registry()
        _settings_list = [widget for widget in _settings_registry.values()]
        return _settings_list
    
    @classmethod
    def apply_user_session(cls, user_session: UserSession):
        """
            Manually apply/reapply user_session to all widgets. Used internally
            by the `on_chat_start` method and should be unnecessary after.
        """
        _settings_registry = cls.settings_registry()
        _ = [widget.apply_user_session(user_session=user_session) for widget in _settings_registry.values()]
        return None
    
    @classmethod
    async def on_settings_update(cls, settings: Dict[str, Any]):
        # callbacks = {widget.key: widget.callback for widget in cls.settings_list()}
        previous_settings = {widget.key: widget.get() for widget in cls.settings_list()}
        print(f"previous_settings: {previous_settings}")
        print(f"settings (changed): {settings}")
        for key, value in settings.items():
            previous_value = previous_settings.get(key, None)
            _settings_widget = cls.retrieve(key)
            _callback = cls.retrieve_callback(key)
            if _settings_widget is not None:
                # if settings widget properly created, change user_session to new value
                _settings_widget.set(value=value)
                # check if callback is defined, if so, call it
                if _callback is not None and previous_value != value:
                    await _callback(previous_value, value)
        return None
    
    @classmethod
    def on_chat_start(cls, cl: chainlit):
        cl.on_settings_update(cls.on_settings_update)
        cls.apply_user_session(user_session=cl.user_session)
        cl.run_sync(cl.ChatSettings(inputs=cls.inputs()).send())

    @classmethod
    def settings(
        cls,
        initial: Optional[T] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
        tooltip: Optional[str] = None,
        type: Optional[InputWidgetType] = "textinput",
        *,
        widget: Optional[InputWidget] = None,
        widget_kwargs: Optional[Dict[str, Any]] = None,
        initial_type: Type[T] = None,
        callback: Optional[Callable[[T, T], Coroutine[Any, Any, None]]] = None,
    ):
        _initial = initial
        _widget = widget
        _label = label
        _description = description
        _tooltip = tooltip
        print(f"settings -> initial: {_initial}") 
        
        def decorator(func: Callable[..., T]):
            print(f"SettingsToolchain -> settings -> decorator")
            print(f"decorator: {decorator}")
            id = func.__name__
            label = _label or func.__name__.title()
            description = _description or _tooltip or func.__doc__
            tooltip = _tooltip or _description or func.__doc__
            initial: T = func() or _initial
            if _widget is None:
                widget = widget_map[type or "textinput"](*[id, label, initial, tooltip, description], **(widget_kwargs or {}))
                print(f"Created widget from widget_map")
            else:
                widget = _widget
                widget.id = id
                widget.label = label if widget.label == "None" else (widget.label or label)
                widget.initial = initial if widget.initial == "None" else (widget.initial or initial)
                widget.tooltip = widget.tooltip or tooltip
                widget.description = widget.description or description
                print(f"Created widget from widget argument")
            
            if widget:
                print(f"registering {widget.id}")
                cls.register(widget)
                cls.register_callback(key=widget.id, callback=callback)
            
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Optional[SettingsToolchain.SettingsWidget[T]]:
                user_sessions = [arg for arg in [*args, *([kwargs.pop("user_session", None)] or [])] if isinstance(arg, UserSession)]
                
                if len(user_sessions) > 0:
                    print(f"wrapper(*args/**kwargs -> {len(user_sessions)} user_sessions) were found.")
                widget_settings = cls.retrieve(func.__name__)
                if widget_settings is not None:
                    _callback = cls.retrieve_callback(widget_settings.key)
                    if _callback is not None:
                        widget_settings.callback = _callback
                if widget_settings is not None and len(user_sessions) > 0:
                    user_session = user_sessions.pop()
                    widget_settings.user_session = user_session
                return widget_settings
            return wrapper
        return decorator
    
    
