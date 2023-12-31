import asyncio
import json
import os
import replicate
import os
import time

from pathlib import Path

from typing import Mapping, Coroutine, Dict, Any, List, Optional, Union, Callable, TypeVar, Type, Literal
from typing_extensions import TypeAlias

from toolchain import ImageToolchain, ImageToolchainTemplates, SessionToolchain
from toolchain.utils import safe_int_parse, parse_replicate_prediction_log
from toolchain.types import (
    ReplicatePredictionStatus, ReplicatePredictionWebhookEventsFilter,
    ParsedReplicatePredictionLog, StableDiffusionVideoParams, STABLE_DIFFUSION_VIDEO,
)

from replicate.prediction import Prediction

from aiohttp import ClientConnectorError
from fastapi import Request, File, UploadFile
from fastapi.responses import HTMLResponse

from chainlit import Action, Message

from chainlit.server import app

import chainlit as cl

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
WEBHOOK_BASE_URL = os.environ.get("WEBHOOK_BASE_URL", "")

_session_toolchain: SessionToolchain = None
_session_toolchain_map: Mapping[str, SessionToolchain] = dict()

_webhook_endpoint = "notice"
get_webhook_url: Callable[[str], str] = lambda public_id: f"{WEBHOOK_BASE_URL}/{_webhook_endpoint}/{public_id}/"

@app.post("/notice/{public_id}/")
async def notice_webhook(request: Request, public_id: str):
    data_json = await request.json()
    prediction: Prediction = None
    try:
        prediction = Prediction(**data_json)
        if prediction is None:
            return {"status": "ok"}
    except Exception as e:
        print("data_json to Prediction failed:", e)
        return {"status": "ok"}
    
    if prediction.status:
        print(prediction.status, end="; ", flush=True)
    else:
        print("status: None", end="; ", flush=True)
    if prediction.output:
        print(prediction.output, end="; ", flush=True)
    
    prediction_succeeded = prediction.status == "succeeded"
    parsed_logs = (
        ParsedReplicatePredictionLog(id=prediction.id, status=prediction.status, percent_complete=100 if prediction_succeeded else 0)
        if prediction.logs is None else
        parse_replicate_prediction_log(id=prediction.id, log_text=prediction.logs, status=prediction.status)
    )
    await replicate_prediction_webhook(public_id=public_id, id=prediction.id, log=parsed_logs, output=prediction.output)
    return {"status": "ok"}


async def replicate_prediction_webhook(
    public_id: str,
    id: str,
    log: ParsedReplicatePredictionLog,
    output: Optional[str] = None,
):
    """Create a new message or update an existing message with the replicate prediction status."""
    global _session_toolchain
    id = id or log.id
    if id is None:
        return print("id and log.id is None")
    # Load session_id
    session_id = _session_toolchain.load_public_session(public_id)
    svd_job = ImageToolchain.from_saved_id(public_id, id)
    
    if svd_job is None:
        return print("svd_job is None for prediction_id: ", id)
    await svd_job.update_progress(
        creation_id=id,
        progress=100 if log.status == "succeeded" or output is not None else log.percent_complete or 0,
        output=output,
    )
    return None


def make_replicate_predictions_params(file_path: Optional[Union[str, Path]] = None, file_url: Optional[str] = None):
    """Create a prediction params `input` dict from the settings."""
    if file_path is None and file_url is None:
        raise ValueError("Must provide either `file_path` or `file_url`.")
    if file_path and file_url:
        raise ValueError("Cannot provide both `file_path` and `file_url`.")
    input_image = file_url or open(file_path, "rb")
    params = StableDiffusionVideoParams(
        # video_length=video_length(),
        # sizing_strategy=sizing_strategy(),
        sizing_strategy="crop_to_16_9",
        # frames_per_second=frames_per_second(),
        # motion_bucket_id=motion_bucket_id(),
        # cond_aug=cond_aug(),
        # decoding_t=decoding_t(),
        # seed=seed(),
    ).to_dict(infer_missing=False)
    params.update({"input_image": input_image})
    return params

async def make_replicate_predictions(
    count: int,
    file_path: Optional[Union[str, Path]] = None,
    file_url: Optional[str] = None,
) -> List[Prediction]:
    global _session_toolchain
    tasks: List[Coroutine[Any, Any, Prediction]] = []
    webhook_events_filter: List[ReplicatePredictionWebhookEventsFilter] = ["start", "output", "logs", "completed"]
    public_id = _session_toolchain.get_public_id()
    webhook_url = get_webhook_url(public_id)
    print("webhook_url: ", webhook_url)
    for _ in range(count):
        for attempt in range(3):  # try 3 times
            try:
                client = replicate.Client(api_token=REPLICATE_API_TOKEN)
                task: Coroutine[Any, Any, Prediction] = client.predictions.async_create(
                    version=STABLE_DIFFUSION_VIDEO.split(":").pop(),
                    input=make_replicate_predictions_params(file_path=file_path, file_url=file_url),
                    webhook=webhook_url,
                    webhook_events_filter=webhook_events_filter,
                    # webhook_completed=webhook_complete_url,
                )
                tasks.append(task)
                break  # if the request is successful, break the loop
            except ClientConnectorError as e:
                if attempt < 2:  # if this is not the last attempt
                    await asyncio.sleep(1)  # wait for 1 second before trying again
                else:  # if this is the last attempt
                    if 'Errno 8' in str(e):
                        # Consider raising a basic exception here as it's auto-converted to `Message`.
                        await cl.Message(content="Connection error. Please refresh the page, select the photo and try again.").send()
                    else:
                        raise  # re-raise the last exception

    return await asyncio.gather(*tasks)




@cl.action_callback(name="Create video")
async def create_video(action: Action):
    """Create a video from images."""
    global _session_toolchain
    public_id = _session_toolchain.get_public_id()
    # image_toolchain = ImageToolchain.from_saved_action(public_id, action)
    image_toolchain = await ImageToolchain.load_from_action(
        public_id=public_id,
        action=action,
        creation_fn=make_replicate_predictions,
        parse_response_fn=lambda prediction: prediction.id,
    )
    print("image_toolchain:\n", image_toolchain, end="\n\n")
    


@cl.on_chat_start
async def on_chat_start():
    global _session_toolchain, _session_toolchain_map
    # TODO: Load minimal session/context, clean up brute force approach
    _session_toolchain = SessionToolchain(cl.user_session)
    session_toolchain = SessionToolchain(cl.user_session)
    public_id = session_toolchain.get_public_id()
    _session_toolchain_map.update({public_id: session_toolchain})


@cl.on_message
async def on_message(message: cl.Message):
    global _session_toolchain
    public_id = _session_toolchain.get_public_id()
    
    content = message.content
    if files := message.elements:
        for file in files:
            if file.type == "image":
                print("public_id", public_id)
                image_toolchain = await ImageToolchain.from_image(
                    public_id=public_id,
                    image_file=file,
                    destination_path=Path("public/images"),
                )

