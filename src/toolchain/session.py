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

_session_map: MutableMapping[SessionId, Type[SessionToolchainType]] = dict()
_session_id_map: MutableMapping[SessionId, str] = dict()

class SessionToolchain:
    _session_id: SessionId = None
    _public_id: str = None
    user_session: UserSession = None
    context: ChainlitContext = None
    
    def _save_session_id(self) -> None:
        """Save a `session_id` with the current user's `public_id`."""
        global _session_id_map
        _session_id_map.update({self.public_id: self.session_id})
        return None
    
    def _save_session(self) -> None:
        """Save a session with the current user's `session_id`."""
        global _session_map
        _session_map.update({self.session_id: self})
        return None
    
    def __init__(self, user_session: UserSession) -> None:
        self.user_session = user_session
        self._save_session_id()
        self._save_session()
        context, _ = self.restore_chainlit_context(self.public_id)
        self.context = context
        return None
    
    @property
    def session_id(self) -> SessionId:
        """Get the session ID from the user session."""
        if self._session_id is None:
            self._session_id = self.user_session.get("id", None)
        return self._session_id
    
    @property
    def public_id(self) -> str:
        """Get the public ID."""
        public_id = self.user_session.get("public_id", None) or self._public_id
        if not public_id:
            public_id = str(uuid.uuid4())
            self.user_session.set("public_id", public_id)
        self._public_id = public_id
        return self._public_id
    
    @staticmethod
    def load_from_session_id(session_id: SessionId) -> Union[SessionToolchainType, None]:
        """Return saved instance of this `SessionToolchain` from `_session_map` with `session_id`."""
        global _session_map
        if session_id is None:
            # Consider raising an exception here.
            return None
        return _session_map.get(session_id, None)
    
    @staticmethod
    def restore_session_id(public_id: str) -> Union[SessionId, None]:
        """Restore a `session_id` from an external connection, ie webhook."""
        global _session_id_map
        if public_id is None:
            # Consider raising an exception here.
            return None
        return _session_id_map.get(public_id, None)
    
    @classmethod
    def from_public_id(cls: Type[SessionToolchainType], public_id: str) -> Union[SessionToolchainType, None]:
        """Return saved instance of this `SessionToolchain` with `public_id`."""
        session_id = cls.restore_session_id(public_id)
        return cls.load_from_session_id(session_id)
    
    @classmethod
    def restore_chainlit_context(cls: Type[SessionToolchainType], public_id: str) -> Tuple[Union[ChainlitContext, None], SessionId]:
        """Restore the `Chainlit` context and the `session_id` from an external connection, ie webhook.

        Args:
            public_id (str): The public ID to load the context with.

        Returns:
            Tuple[Union[ChainlitContext, None], SessionId]: A tuple containing the context and the session ID."""
        session_id = cls.restore_session_id(public_id)
        context: ChainlitContext = None
        if session_id:
            ws_session = WebsocketSession.get_by_id(session_id=session_id)
            context = init_ws_context(ws_session)
        return context, session_id
    
    @classmethod
    def restore_chainlit_context_session_id(cls: Type[SessionToolchainType], public_id: str) -> SessionId:
        """Restore the `Chainlit` context and the `session_id` from an external connection, ie webhook.

        Args:
            public_id (str): The public ID to load the session from.

        Returns:
            str: The session ID."""
        _, session_id = cls.restore_chainlit_context(public_id)
        return session_id



