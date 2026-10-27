import json
import re

from huggingface_hub import InferenceClient

from app.config import (
    HF_MODEL,
    HF_PROVIDER,
    HF_TOKEN,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)
from app.prompts import SYSTEM_PROMPT


class LLMClient:
    def __init__(self) -> None:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set. Create a Hugging Face access token "
                "with Inference Providers permission and put it in .env."
            )

        self.client = InferenceClient(
            provider=HF_PROVIDER,
            api_key=HF_TOKEN,
        )

    @staticmethod
    def _extract_json(content: str) -> dict:
        cleaned = content.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    "The Hugging Face model did not return valid JSON.\n"
                    f"Raw response:\n{content}"
                )

            try:
                return json.loads(
                    cleaned[start : end + 1]
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "The Hugging Face model returned malformed JSON.\n"
                    f"Raw response:\n{content}"
                ) from exc

    def generate_profile(self, user_prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=2500,
                temperature=TEMPERATURE,

                # Provider/model-specific parameters go here.
                extra_body={
                    "reasoning_effort": "low"
                },
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not call Hugging Face Inference Providers "
                f"with model '{HF_MODEL}' and provider "
                f"'{HF_PROVIDER}'. Original error: {exc}"
            ) from exc

        message = response.choices[0].message

        content = getattr(message, "content", None)

        if not content:
            raise RuntimeError(
                "Hugging Face did not return final answer content.\n"
                f"finish_reason: "
                f"{response.choices[0].finish_reason}\n"
                f"message: {message}"
            )

        return self._extract_json(content)