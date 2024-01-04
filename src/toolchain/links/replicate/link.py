import asyncio
import os
import replicate

from typing import (
    Callable, Coroutine, List, Dict, Any, Optional, Literal, Tuple, Type, TypeVar, Union, Mapping,
    BinaryIO, TextIO, Awaitable,
)
from typing_extensions import TypeAlias

from pathlib import Path

from fastapi import FastAPI

from replicate.prediction import Prediction

from aiohttp import ClientConnectorError
from fastapi import Request, File, UploadFile
from fastapi.responses import HTMLResponse

from chainlit import Action, Message
from chainlit.user_session import UserSession

from toolchain import SessionToolchain, ImageToolchain
from toolchain.links.replicate.models import ParsedPredictionLog, ReplicateModel, StableDiffusionVideoParams, PlaygroundV2_1024pxAestheticParams
from toolchain.links.replicate.types import (
    PredictionStatus, PredictionWebhookEventsFilter, STABLE_VIDEO_DIFFUSION_MODEL, PLAYGROUND_V2_1024PX_AESTHETIC,
)


WEBHOOK_URL_TEMPLATE = "{BASE_URL}/{ENDPOINT}/{PUBLIC_ID}/"

class ReplicateLink:
    app: FastAPI = None
    session_toolchain: Optional[SessionToolchain] = None
    image_toolchain: Optional[ImageToolchain] = None
    api_key: str = None
    webhook_base_url: str = None
    webhook_endpoint: str = None
    on_webhook: Callable[[str, ParsedPredictionLog, Prediction], None] = None
    on_webhook_async: Callable[[str, ParsedPredictionLog, Prediction], Coroutine[Any, Any, None]] = None
    session_id: str = None
    public_id: str = None
    
    prediction_ids: List[str] = []
    
    # models
    stable_video_diffusion = ReplicateModel(STABLE_VIDEO_DIFFUSION_MODEL)
    playground_v2_1024px_aesthetic = ReplicateModel(PLAYGROUND_V2_1024PX_AESTHETIC)
    
    
    def __init__(
        self,
        app: FastAPI,
        session_toolchain: Optional[SessionToolchain] = None,
        *,
        api_key: str = None,
        webhook_base_url: str = None,
        webhook_endpoint: str = None,
        on_webhook: Callable[[str, ParsedPredictionLog, Prediction], None] = None,
        on_webhook_async: Callable[[str, ParsedPredictionLog, Prediction], Coroutine[Any, Any, None]] = None,
    ):
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
        self.webhook_base_url = webhook_base_url or os.environ.get("WEBHOOK_BASE_URL", "")
        self.webhook_endpoint = webhook_endpoint or os.environ.get("WEBHOOK_ENDPOINT", "notice")
        self.on_webhook = on_webhook
        self.on_webhook_async = on_webhook_async
        self.app = self._create_endpoint(app)
        self.session_toolchain = session_toolchain
        return None
    
    async def _handle_webhook(self, public_id: str, parsed_logs: ParsedPredictionLog, prediction: Prediction):
            """
            Handles the Replicate webhook, passing the `public_id`, parsed logs and raw `prediction`
            to any registered `on_webhook` callback.

            Args:
                public_id (str): The public ID associated with the user.
                parsed_logs (ParsedPredictionLog): The parsed Replicate `Prediction` log.
                prediction (Prediction): The `Prediction` response from Replicate.

            Returns:
                None
            """
            if self.on_webhook_async:
                await self.on_webhook_async(public_id, parsed_logs, prediction)
            elif self.on_webhook:
                self.on_webhook(public_id, parsed_logs, prediction)
            return None
    
    def _create_endpoint(self, app: FastAPI):
        """
        Creates an endpoint for handling webhooks.

        Args:
            app (FastAPI): The FastAPI instance.

        Returns:
            FastAPI: The FastAPI instance with the webhook endpoint added.
        """
        endpoint_path = "/" + "/".join([self.webhook_endpoint, "{public_id}"]) + "/"
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
                print(data_json)
                return {"status": "ok"}
            
            parsed_logs = ParsedPredictionLog.parse(prediction=prediction)
            await self._handle_webhook(public_id, parsed_logs, prediction)
            return {"status": "ok"}
        return app
    
    def on_chat_start(
        self,
        action_callback: Callable,
        session_toolchain: Optional[SessionToolchain] = None,
        user_session: Optional[UserSession] = None,
    ):
        """Simple, consistant way to load session across Toolchain and Link classes."""
        if session_toolchain is None and user_session is None:
            raise ValueError("Either session_toolchain or user_session must be provided.")
        if session_toolchain is None:
            session_toolchain = SessionToolchain(user_session=user_session)
        self.public_id = session_toolchain.get_public_id()
        self.session_id = session_toolchain.session_id
        self.session_toolchain = session_toolchain
    
        async def create_video(action: Action):
            session = SessionToolchain.from_public_id(public_id=self.public_id)
            image_toolchain = await ImageToolchain.load_from_creation_action(
                public_id=self.public_id,
                action=action,
                creation_fn=self.make_stable_video_diffusion_creation_fn(),
                parse_response_fn=lambda response: response.id,
            )
            if image_toolchain is None:
                print("ReplicateLink -> create_video -> Error loading image toolchain")
                return
            return
        action_callback("Create video")(create_video)
        
        async def create_image(action: Action):
            session = SessionToolchain.from_public_id(public_id=self.public_id)
            image_toolchain = await ImageToolchain.load_from_creation_action(
                public_id=self.public_id,
                action=action,
                creation_fn=self.make_playground_v2_1024px_aesthetic_creation_fn(),
                parse_response_fn=lambda response: response.id,
            )
            if image_toolchain is None:
                print("ReplicateLink -> create_image -> Error loading image toolchain")
                return
            return
        action_callback("Create image")(create_image)
        
        async def on_webhook_async(
            public_id: str,
            parsed_logs: ParsedPredictionLog,
            prediction: Prediction,
        ):
            session = SessionToolchain.from_public_id(public_id=public_id)
            image_toolchain = ImageToolchain.load_from_image_path_map_by_id(
                public_id=public_id, creation_id=prediction.id,
            )
            if image_toolchain is None:
                print("ReplicateLink -> on_webhook_async -> Error loading image toolchain")
                return
            await image_toolchain.update_progress(
                creation_id=prediction.id,
                progress=parsed_logs.percent_complete or 0,
                output=prediction.output,
            )
            return None
        self.on_webhook_async = on_webhook_async
        
        return None
    
    async def on_message(self, message: Message, destination_path: Optional[Path] = None):
        images = [el for el in message.elements if el.type == "image"]
        session = SessionToolchain.from_public_id(public_id=self.public_id)
        
        if len(message.content) > 3 and not message.content.isnumeric():
            self.image_toolchain = await ImageToolchain.from_prompt(
                public_id=self.public_id,
                prompt=message.content,
                destination_path=destination_path or Path("public/images"),
            )
        elif len(images) > 0:
            self.image_toolchain = await asyncio.gather(*[
                ImageToolchain.from_image(
                    public_id=self.public_id,
                    image_file=image_file,
                    destination_path=destination_path or Path("public/images"),
                ) for image_file in images
            ])
    
    @property
    def client(self):
        return replicate.Client(api_token=self.api_key)
    
    @property
    def webhook_url(self):
        """The full webhook URL as should be registered with Replicate."""
        url = WEBHOOK_URL_TEMPLATE.format(
            BASE_URL=self.webhook_base_url,
            ENDPOINT=self.webhook_endpoint,
            PUBLIC_ID=self.public_id,
        )
        return url
    
    async def get_prediction(self, id: str) -> Prediction:
        """
        Get the latest prediction for the given ID.

        Parameters:
            id (str): The ID of the prediction.

        Returns:
            Prediction: The latest prediction for the given ID.
        """
        return await self.client.predictions.get(id)
    
    async def get_predictions(self, ids: Optional[List[str]] = None) -> Prediction:
        """Get the latest predictions for the given IDs.

        Args:
            ids (Optional[List[str]]): The list of IDs for which to retrieve predictions.
                If not provided, previously created predictions' ids will be used.

        Returns:
            Prediction: The latest predictions for the given IDs.
        """
        ids = ids or self.prediction_ids
        return await asyncio.gather(*[self.get_prediction(id) for id in ids])
    
    async def get_status(self, id: str) -> str:
        """
        Get the status of the prediction for the given ID.

        Parameters:
            id (str): The ID of the prediction.

        Returns:
            str: The status of the prediction for the given ID.
        """
        prediction = await self.get_prediction(id=id)
        return prediction.status
    

    def make_stable_video_diffusion_creation_fn(
        self,
        params: Optional[Union[StableDiffusionVideoParams, List[StableDiffusionVideoParams]]] = None,
        webhook_events_filter: List[PredictionWebhookEventsFilter] = ["start", "output", "logs", "completed"],
    ) -> Callable[[int, Union[str, Path, List[str], List[Path]]], Awaitable[List[Prediction]]]:
        """Helper function to make the required `creation_fn` used by `ImageToolchain`."""
        if params is None or len(params or []) == 0:
            params = StableDiffusionVideoParams()
        async def creation_fn(
            count: int,
            file_path: Optional[Union[str, Path]] = None,
            file_url: Optional[str] = None,
        ) -> List[Prediction]:
            # safety precaution: resize parameters to match count; allow loosely different parameters on each creation
            parameters = params if isinstance(params, list) else [params]
            parameters = [parameters[i % len(parameters)] for i in range(count)]
            if file_path is None and file_url is None:
                raise ValueError("Either file_path or file_url must be provided.")
            tasks: List[Coroutine[Any, Any, Prediction]] = []
            predictions: List[Prediction] = []
            for i in range(count):
                for attempt in range(3):  # try 3 times
                    try:
                        if file_path:
                            input_image = open(file_path, "rb")
                        elif file_url:
                            input_image = file_url
                        params_set = parameters[i].to_dict(infer_missing=False)
                        params_set.update({"input_image": input_image})
                        task: Coroutine[Any, Any, Prediction] = self.client.predictions.async_create(
                            version=self.stable_video_diffusion.version,
                            input=params_set,
                            webhook=self.webhook_url,
                            webhook_events_filter=webhook_events_filter,
                        )
                        tasks.append(task)
                        break  # break out of successfull attempt loop
                    except ClientConnectorError as e:
                        if attempt < 2: # if this is not the last attempt
                            await asyncio.sleep(1) # wait for 1 second before trying again
                        else:  # if this is the last attempt
                            if 'Errno 8' in str(e):
                                # Consider raising a basic exception here as it's auto-converted to `Message`.
                                raise NameError("Connection error. Please refresh the page, select the photo and try again.")
                            else:
                                raise  # re-raise the last exception
            predictions = await asyncio.gather(*tasks)
            self.prediction_ids.extend([p.id for p in predictions])
            return predictions
        return creation_fn
    
    def make_playground_v2_1024px_aesthetic_creation_fn(
        self,
        params: Optional[Union[PlaygroundV2_1024pxAestheticParams, List[PlaygroundV2_1024pxAestheticParams]]] = None,
        webhook_events_filter: List[PredictionWebhookEventsFilter] = ["start", "output", "logs", "completed"],
    ):
        if params is None or len(params or []) == 0:
            params = PlaygroundV2_1024pxAestheticParams()
        async def creation_fn(
            count: int,
            prompt: str,
            negative_prompt: str = "",
        ):
            parameters = params if isinstance(params, list) else [params]
            parameters = [parameters[i % len(parameters)] for i in range(count)]
            tasks: List[Coroutine[Any, Any, Prediction]] = []
            predictions: List[Prediction] = []
            for i in range(count):
                for attempt in range(3):
                    try:
                        params_set = parameters[i].to_dict(infer_missing=False)
                        params_set.update({"prompt": prompt, "negative_prompt": negative_prompt})
                        task: Coroutine[Any, Any, Prediction] = self.client.predictions.async_create(
                            version=self.playground_v2_1024px_aesthetic.version,
                            input=params_set,
                            webhook=self.webhook_url,
                            webhook_events_filter=webhook_events_filter,
                        )
                        tasks.append(task)
                        break
                    except ClientConnectorError as e:
                        if attempt < 2:
                            await asyncio.sleep(1)
                        else:
                            if 'Errno 8' in str(e):
                                raise NameError("Connection error. Please refresh the page, enter a new prompt and try again.")
                            else:
                                raise
            predictions = await asyncio.gather(*tasks)
            self.prediction_ids.extend([p.id for p in predictions])
            return predictions
        return creation_fn


