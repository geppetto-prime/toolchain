"""This module contains the types used by the toolchains."""
import dataclasses

from typing import Optional, Literal
from typing_extensions import TypeAlias

from pathlib import Path

from pydantic import BaseModel, Field, root_validator, ConfigDict
from pydantic.dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin



# pyautogen[retrievechat] added:
Extensions: TypeAlias = Literal['txt', 'json', 'csv', 'tsv', 'md', 'html', 'htm', 'rtf', 'rst', 'jsonl', 'log', 'xml', 'yaml', 'yml', 'pdf']

# unstructured[all-docs] added:
ExtensionsExtended: TypeAlias = Literal['docx', 'doc', 'odt', 'pptx', 'ppt', 'xlsx', 'eml', 'msg', 'epub']



_IMAGE_FILE_LINK_UNSAFE_HTML = """<span class=""><a href="{URL}" target="_blank">{NAME}</a></span>"""
_IMAGE_DISPLAY_LINK_UNSAFE_HTML = """<div class="cl-image-container" style="max-width:{MAX_WIDTH};"><a href="{URL}" target="_blank"><img src="{URL}"></a></div>"""
_IMAGE_CONTAINER_UNSAFE_HTML = """\
<div>
    <div class="cl-description-container cl-force-wrap">
        <strong style="display: block;">{TITLE}</strong>
        {IMAGE_FILE_LINK}
    </div>
    {IMAGE_DISPLAY_LINK}
</div>"""

@dataclass
class ImageToolchainTemplates:
    """
    A class that represents the templates used in the image toolchain.

    Attributes:
        unsafe_html (bool): Flag indicating whether unsafe HTML is allowed.
     """

    unsafe_html: bool = dataclasses.field(default=True)
    _FILE_NAME_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="{PREFIX}_{TIMESTAMP}.{EXT}", init=False, kw_only=True)
    _FILE_NAME_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="{PREFIX}_{TIMESTAMP}{EXT}", init=False, kw_only=True)
    _PROMPT_FILE_NAME_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="{PROMPT}_{TIMESTAMP}{EXT}", init=False, kw_only=True)
    _PROMPT_FILE_NAME_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="{PROMPT}_{TIMESTAMP}{EXT}", init=False, kw_only=True)
    _CREATE_FROM_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="**Create {TYPE} from:**\n<span class='cl-force-wrap'>{FROM}</span>", init=False, kw_only=True)
    _CREATE_FROM_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="**Create {TYPE} from:**\n{FROM}", init=False, kw_only=True)
    _CREATING_COUNT_FROM_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="**Creating <span class='cl-red-text'>{COUNT}</span> {TYPE}{PLURAL} from:**\n<span class='cl-force-wrap'>{IMAGE_NAME}</span>", init=False, kw_only=True)
    _CREATING_COUNT_FROM_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="**Creating {COUNT} {TYPE}{PLURAL} from:**\n{IMAGE_NAME}", init=False, kw_only=True)
    _DOWNLOAD_FILE_MARKDOWN_TEMPLATE_UNSAFE_HTML: Optional[str] = dataclasses.field(default="""**{TYPE} generation complete.**\n<span class='cl-force-wrap'>Download [{FILE_NAME}]({FILE_URL})</span>""", init=False, kw_only=True)
    _DOWNLOAD_FILE_MARKDOWN_TEMPLATE_SAFE_HTML: Optional[str] = dataclasses.field(default="""**{TYPE} generation complete.**\nDownload [{FILE_NAME}]({FILE_URL})""", init=False, kw_only=True)
    _IMAGE_CONTAINER_UNSAFE_HTML: Optional[str] = dataclasses.field(default="""<div class="cl-image-container" style="max-width: {MAX_WIDTH};"><a href="{URL}" target="_blank"><img src="{URL}" class=""></a></div>""", init=False, kw_only=True)
    _IMAGE_CONTAINER_SAFE_HTML: Optional[str] = dataclasses.field(default="""[{MAX_WIDTH}]({URL})""", init=False, kw_only=True)

    @property
    def FILE_NAME_TEMPLATE(self) -> str:
        """Get the file name template based on the safety of HTML.

        Returns:
            str: The file name template."""
        return self._FILE_NAME_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._FILE_NAME_TEMPLATE_SAFE_HTML
    
    @property
    def PROMPT_FILE_NAME_TEMPLATE(self) -> str:
        """Get the file name template for prompt creations based on the safety of HTML.

        Returns:
            str: The file name template for prompt images."""
        return self._PROMPT_FILE_NAME_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._PROMPT_FILE_NAME_TEMPLATE_SAFE_HTML
        

    @property
    def CREATE_FROM_TEMPLATE(self) -> str:
        """Get the template describing what `TYPE` ie: 'video' to create based on the safety of HTML.

        Returns:
            str: The template describing what to create."""
        return self._CREATE_FROM_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._CREATE_FROM_TEMPLATE_SAFE_HTML

    @property
    def CREATING_COUNT_FROM_TEMPLATE(self) -> str:
        """Get the template describing what `TYPE` ie: 'video' currently being created based on the safety of HTML.

        Returns:
            str: The template describing what is currently being created."""
        return self._CREATING_COUNT_FROM_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._CREATING_COUNT_FROM_TEMPLATE_SAFE_HTML

    @property
    def DOWNLOAD_FILE_MARKDOWN_TEMPLATE(self) -> str:
        """Get the template for downloading files based on the safety of HTML.

        Returns:
            str: The template for downloading files."""
        return self._DOWNLOAD_FILE_MARKDOWN_TEMPLATE_UNSAFE_HTML if self.unsafe_html else self._DOWNLOAD_FILE_MARKDOWN_TEMPLATE_SAFE_HTML
    
    @property
    def IMAGE_CONTAINER(self) -> str:
        """Get the template for displaying images based on the safety of HTML.
        'Safe' option is not usable because the `MAX_WIDTH` value is not able to be hidden,
        instead it is displayed as text.

        Returns:
            str: The template for displaying images."""
        return self._IMAGE_CONTAINER_UNSAFE_HTML if self.unsafe_html else self._IMAGE_CONTAINER_SAFE_HTML
    
    def render_image_container(
        self,
        title: str,
        image_name: str,
        image_url: str,
        display_image: bool = True,
        max_width: str = "50%",
    ) -> str:
        """Get the template for displaying images based on the safety of HTML."""
        if not self.unsafe_html:
            return f"**{title}**\n[{image_name}]({image_url})"
        image_file_link = _IMAGE_FILE_LINK_UNSAFE_HTML.format(URL=image_url, NAME=image_name)
        image_display_link = _IMAGE_DISPLAY_LINK_UNSAFE_HTML.format(MAX_WIDTH=max_width, URL=image_url)
        image_container = _IMAGE_CONTAINER_UNSAFE_HTML.format(
            TITLE=title,
            IMAGE_FILE_LINK=image_file_link,
            IMAGE_DISPLAY_LINK=image_display_link if display_image else "",
        )
        return image_container



