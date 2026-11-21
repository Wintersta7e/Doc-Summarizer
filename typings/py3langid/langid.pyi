# SPDX-License-Identifier: MIT
from collections.abc import Sequence
from typing import Self

MODEL_FILE: str

class LanguageIdentifier:
    @classmethod
    def from_model_file(cls, model_file: str, *, norm_probs: bool = ...) -> Self: ...
    def set_languages(self, langs: Sequence[str] | None = None) -> None: ...
    def classify(self, text: str) -> tuple[str, float]: ...
