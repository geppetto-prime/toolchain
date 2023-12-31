import uuid

from typing import Optional, Tuple, Mapping, TypeVar, Type

import chainlit


from chainlit.context import init_http_context, init_ws_context, ChainlitContext
from chainlit.session import HTTPSession, WebsocketSession
from chainlit.user_session import UserSession

from toolchain.models import DataClassJsonMixinPro

_session_map: Mapping[str, type['SessionToolchain']] = dict()
_session_id_map: Mapping[str, str] = dict()
PUBLIC_ID_KEY = "public_id"

Session = TypeVar('Session', bound="SessionToolchain")

class SessionToolchain:
    public_id: str = None
    user_session: UserSession = None
    
    def __init__(self, user_session: UserSession) -> None:
        global _session_map, _session_id_map
        self.user_session = user_session
        self.refresh_public_id()
        self.save_session()

    # def save_session(self, session: Type[Session]) -> Session:
    def save_session(self) -> Session:
        """Save a session to the current user's `public_id`."""
        global _session_map, _session_id_map
        self.refresh_public_id()
        _session_map.update({self.public_id: self})
        _session_id_map.update({self.public_id: self.session_id()})
        return self

    def load_session(public_id: str) -> Session:
        """Load a session from the current user's `public_id`."""
        global _session_map
        return _session_map.get(public_id, None)

    def load_session_id(public_id: str) -> Session:
        """Load a session from the current user's `public_id`."""
        global _session_id_map
        return _session_id_map.get(public_id, None)

    def session_id(self) -> str:
        return self.user_session.get("id", "None")

    def refresh_public_id(self, value: Optional[str] = None) -> str:
        """Get or set the public id for the current user used for webhooks."""
        if value:
            self.user_session.set(PUBLIC_ID_KEY, value)
        value = self.user_session.get(PUBLIC_ID_KEY, None)
        if not value:
            value = str(uuid.uuid4())
            self.user_session.set(PUBLIC_ID_KEY, value)
        self.public_id = value
        return value

    def get_public_id(self, value: Optional[str] = None) -> str:
        """Get or set the public id for the current user used for webhooks."""
        return self.public_id or self.refresh_public_id(value)

    def register_public_session(self, public_id: Optional[str] = None) -> str:
        """Register a public id for the current user used for webhooks."""
        global _session_map, _session_id_map
        self.public_id = public_id or self.refresh_public_id()
        self.save_session()
        # _session_map.update({public_id: self.session_id()})
        return public_id

    def load_public_session_context(self, public_id: str) -> Tuple[str, Optional[ChainlitContext]]:
        """Restore a `session_id` from an external connection, ie webhook."""
        global _session_map, _session_id_map
        session_id = _session_id_map.get(public_id, None)
        context: ChainlitContext = None
        if session_id:
            ws_session = WebsocketSession.get_by_id(session_id=session_id)
            context = init_ws_context(ws_session)
        return session_id, context

    def load_public_session(self, public_id: str) -> str:
        """Restore a `session_id` from an external connection, ie webhook."""
        session_id, _ = self.load_public_session_context(public_id)
        return session_id



