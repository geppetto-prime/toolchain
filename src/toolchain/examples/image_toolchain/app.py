import asyncio
import os
import replicate

from pathlib import Path

from typing import Coroutine, Any, List, Optional, Union, Callable

from replicate.prediction import Prediction

from fastapi import Request

from chainlit import Action, Message
from chainlit.element import Element

from chainlit.server import app

import chainlit as cl

from toolchain.links.replicate.models import ParsedPredictionLog

from toolchain import ImageToolchain, SessionToolchain


STABLE_VIDEO_DIFFUSION_MODEL: str = "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
WEBHOOK_BASE_URL = os.environ.get("WEBHOOK_BASE_URL", "")
WEBHOOK_ENDPOINT = os.environ.get("WEBHOOK_ENDPOINT", "notice")
get_webhook_url: Callable[[str], str] = lambda public_id: f"{WEBHOOK_BASE_URL}/{WEBHOOK_ENDPOINT}/{public_id}/"


endpoint_path = "/" + "/".join([WEBHOOK_ENDPOINT, "{public_id}"]) + "/"
@app.post(endpoint_path)
async def on_webhook(request: Request, public_id: str):
    """
    Endpoint registered with Replicate.

    Args:
        request (Request): The incoming request object.
        public_id (str): The public ID associated with the `Chainlit` user.
    """
    data_json = await request.json()
    prediction: Prediction = None
    try:
        prediction = Prediction(**data_json)
        if prediction is None:
            return {"status": "ok"}
    except Exception as e:
        print("data_json to Prediction failed:", e)
        return {"status": "ok"}
    
    prediction_succeeded = prediction.status == "succeeded"
    parsed_logs = (
        ParsedPredictionLog(id=prediction.id, status=prediction.status, percent_complete=100 if prediction_succeeded else 0)
        if prediction.logs is None else
        ParsedPredictionLog.parse(id=prediction.id, log_text=prediction.logs, status=prediction.status)
    )
    await on_webhook_async(public_id, parsed_logs, prediction)
    return {"status": "ok"}



async def on_webhook_async(
    public_id: str,
    parsed_logs: ParsedPredictionLog,
    prediction: Prediction,
):
    image_toolchain = ImageToolchain.load_from_image_path_map_by_id(
        public_id=public_id, creation_id=prediction.id
    )
    if image_toolchain is None:
        print("Error loading image toolchain")
        return None
    await image_toolchain.update_progress(
        creation_id=prediction.id,
        progress=parsed_logs.percent_complete or 0,
        output=prediction.output,
    )
    return


stable_video_diffusion_params = {
    "video_length": "25_frames_with_svd_xt",
    "sizing_strategy": "maintain_aspect_ratio",
    "frames_per_second": 6,
    "motion_bucket_id": 127,
    "cond_aug": 0.02,
    "decoding_t": 14,
}


def make_creation_fn(
    public_id: str,
    webhook_events_filter: List[str] = ["start", "output", "logs", "completed"],
):
    async def creation_fn(
        count: int,
        file_path: Optional[Union[str, Path, List[str], List[Path]]] = None,
    ) -> List[Prediction]:
        tasks: List[Coroutine[Any, Any, Prediction]] = []
        for _ in range(count):
            input_image = open(file_path, "rb")
            stable_video_diffusion_params.update({"input_image": input_image})
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            task: Coroutine[Any, Any, Prediction] = client.predictions.async_create(
                version=STABLE_VIDEO_DIFFUSION_MODEL.split(":").pop(),
                input=stable_video_diffusion_params,
                webhook=get_webhook_url(public_id=public_id),
                webhook_events_filter=webhook_events_filter,
            )
            tasks.append(task)
        predictions = await asyncio.gather(*tasks)
        return predictions
    return creation_fn


@cl.action_callback("Create video")
async def create_video(action: Action):
    public_id = SessionToolchain(user_session=cl.user_session).get_public_id()
    image_toolchain = await ImageToolchain.load_from_creation_action(
        public_id=public_id,
        action=action,
        creation_fn=make_creation_fn(public_id=public_id),
        parse_response_fn=lambda response: response.id,
    )
    return


@cl.on_message
async def on_message(message: Message):
    print("on_message")
    
    # get public ID from session
    public_id = SessionToolchain(user_session=cl.user_session).get_public_id()
    
    # select images from message
    images: List[Element] = [el for el in message.elements if el.type == "image"]
    
    # create image toolchains for each image attachment
    image_toolchains = await asyncio.gather(*[
        ImageToolchain.from_image(
            public_id,
            image_file,
            destination_path=Path("public/images"),
        ) for image_file in images
    ])





@cl.on_chat_start
async def on_chat_start():
    print("on_chat_start")
    session = SessionToolchain(user_session=cl.user_session)
    










