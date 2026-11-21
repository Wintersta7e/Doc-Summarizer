# SPDX-License-Identifier: MIT
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal, NotRequired, TypedDict, overload

class ChatCompletionResponseMessage(TypedDict):
    role: str
    content: str | None

class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str | None

class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CreateChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: NotRequired[CompletionUsage]

class ChatCompletionStreamResponseDelta(TypedDict, total=False):
    role: str
    content: str | None

class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: str | None

class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionStreamResponseChoice]

def llama_supports_gpu_offload() -> bool: ...

class Llama:
    n_tokens: int
    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = ...,
        n_threads: int | None = ...,
        n_threads_batch: int | None = ...,
        n_gpu_layers: int = ...,
        verbose: bool = ...,
    ) -> None: ...
    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[False] = False,
        **kwargs: object,
    ) -> CreateChatCompletionResponse: ...
    @overload
    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        stream: Literal[True],
        **kwargs: object,
    ) -> Iterator[CreateChatCompletionStreamResponse]: ...
    def tokenize(self, text: bytes, *, add_bos: bool = True) -> list[int]: ...
    def reset(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...
