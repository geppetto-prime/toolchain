
import asyncio

from pathlib import Path

from chainlit.server import app

import chainlit as cl


from toolchain.links.replicate.link import ReplicateLink

replicate_link = ReplicateLink(app=app)



@cl.on_message
async def on_message(message: cl.Message):
    await replicate_link.on_message(message=message)


@cl.on_chat_start
async def on_chat_start():
    replicate_link.on_chat_start(
        action_callback=cl.action_callback,
        user_session=cl.user_session,
    )
















