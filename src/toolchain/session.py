"""Session management for Chainlit context."""
import uuid

from typing import Optional, Tuple, Mapping, MutableMapping, TypeVar, Type, Union

import chainlit


from chainlit.context import init_http_context, init_ws_context, ChainlitContext
from chainlit.session import HTTPSession, WebsocketSession
from chainlit.user_session import UserSession

from toolchain.models import DataClassJsonMixinPro

SessionToolchainType = TypeVar('SessionToolchainType', bound="SessionToolchain")
SessionId = TypeVar('SessionId', bound=str)

class SessionToolchain:
    session_id: SessionId = None
    _public_id_session_map: MutableMapping[str, SessionToolchainType] = dict()
    user_session: UserSession = None
    
    def __init__(
        self,
        user_session: UserSession,
    ) -> None:
        self.user_session = user_session
        self._on_init_save()
        return None
    
    def _set_public_id(self, public_id: str) -> None:
        """Manually set the public ID. Useful when restoring context from webhook."""
        self.user_session.set("public_id", public_id)
        return None
    
    def get_public_id(self) -> str:
        """Get the public ID."""
        def generate_id() -> str:
            return str(uuid.uuid4()).replace("-", "")
        pid = self.user_session.get("public_id", generate_id())
        self._set_public_id(pid)
        return pid
    
    def get_session_id(self) -> SessionId:
        """Get the session ID from the user session."""
        return self.user_session.get("id", None)
    
    def _on_init_save(self, overwrite: bool = False):
        """Save the session."""
        public_id = self.get_public_id()
        self.session_id = self.get_session_id()
        if public_id not in SessionToolchain._public_id_session_map or overwrite:
            SessionToolchain._public_id_session_map.update({public_id: self})
        return None
    
    @staticmethod
    def _load_context(session_id: SessionId) -> ChainlitContext:
        """Load a context."""
        ws_session = WebsocketSession.get_by_id(session_id=session_id)
        context: ChainlitContext = init_ws_context(ws_session)
        return context
    
    @classmethod
    def _load_instance(cls: Type[SessionToolchainType], public_id: str) -> SessionToolchainType:
        """Load a session.
        Warning: Must also restore the `ChainlitContext` with `_load_context`."""
        instance: SessionToolchainType = cls._public_id_session_map.get(public_id, None)
        return instance
    
    @classmethod
    def from_public_id(cls: Type[SessionToolchainType], public_id: str) -> SessionToolchainType:
        """Restore a SessionToolchain from a public_id.
        Required for loading a SessionToolchain from a webhook."""
        instance: SessionToolchainType = cls._load_instance(public_id)
        if instance is None:
            public_ids = "; ".join(list(cls._public_id_session_map.keys()))
            print(f"SessionToolchain {public_id} not found in:\n{public_ids}\n")
            return None
        instance._load_context(instance.session_id)
        instance._set_public_id(public_id)
        return instance



