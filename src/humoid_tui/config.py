from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ProviderName = Literal[
    "digitalocean",
    "meta",
    "openai",
    "llamacpp",
    "litert",
]

MemoryBackendName = Literal[
    "auto",
    "weaviate",
    "embedded-weaviate",
    "sqlite",
]

Gemma4Mode = Literal[
    "auto",
    "native",
    "off",
]

Gemma4ThinkingMode = Literal[
    "off",
    "low",
    "on",
]

AutonomyMode = Literal[
    "review",
    "balanced",
    "autonomous",
]

ToolProtocolName = Literal[
    "auto",
    "openai",
    "gemma4",
    "tagged-json",
    "humoid-v1",
]


class ProviderConfig(BaseModel):
    name: ProviderName
    api_key: str
    base_url: str
    model: str

    timeout_seconds: float = 180.0
    max_retries: int = 3

    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, object] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Main provider
    # ------------------------------------------------------------------

    humoid_provider: ProviderName = "digitalocean"
    humoid_language: str = "en"

    # ------------------------------------------------------------------
    # DigitalOcean Gradient AI
    #
    # Use the SDK base URL here.
    # The OpenAI client appends /chat/completions automatically.
    # ------------------------------------------------------------------

    digitalocean_api_key: str = ""
    digitalocean_base_url: str = "https://inference.do-ai.run/v1"
    digitalocean_model: str = ""

    digitalocean_timeout_seconds: float = 180.0
    digitalocean_max_retries: int = 3

    # ------------------------------------------------------------------
    # Meta Model API / Muse Spark
    # ------------------------------------------------------------------

    meta_api_key: str = ""
    meta_base_url: str = ""
    meta_model: str = "muse-spark-1.1"

    meta_enable_search_grounding: bool = False
    meta_timeout_seconds: float = 180.0
    meta_max_retries: int = 3

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.6"

    openai_timeout_seconds: float = 180.0
    openai_max_retries: int = 3

    # ------------------------------------------------------------------
    # Local llama.cpp server
    # ------------------------------------------------------------------

    llamacpp_api_key: str = "local"
    llamacpp_base_url: str = "http://127.0.0.1:8080/v1"
    llamacpp_model: str = "local-model"

    llamacpp_timeout_seconds: float = 300.0
    llamacpp_max_retries: int = 1

    # ------------------------------------------------------------------
    # LiteRT-LM OpenAI-compatible shim
    # ------------------------------------------------------------------

    litert_api_key: str = "local"
    litert_base_url: str = "http://127.0.0.1:8090/v1"
    litert_model: str = "litert-local"

    litert_timeout_seconds: float = 300.0
    litert_max_retries: int = 1

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    humoid_workspace: Path = Path("./workspace")

    humoid_autonomy_mode: AutonomyMode = "balanced"

    humoid_allow_shell: bool = False
    humoid_shell_timeout_seconds: int = 45

    humoid_max_tool_rounds: int = 12
    humoid_max_output_tokens: int = 6000
    humoid_temperature: float = 0.2

    humoid_context_limit: int = 20000
    humoid_gemma_context_limit: int = 10000
    humoid_glm_context_limit: int = 20000
    humoid_show_reasoning: bool = True

    # ------------------------------------------------------------------
    # Tool protocol system
    # ------------------------------------------------------------------

    humoid_tool_protocol: ToolProtocolName = "auto"
    humoid_parallel_tool_calls: bool = True

    humoid_tool_argument_max_bytes: int = 131072
    humoid_tool_result_max_bytes: int = 200000

    humoid_tool_retry_limit: int = 2
    humoid_tool_loop_repeat_limit: int = 3

    humoid_require_tool_schema_validation: bool = True
    humoid_require_tool_name_validation: bool = True
    humoid_enable_tool_json_repair: bool = True

    # ------------------------------------------------------------------
    # Gemma 4 native tool-calling settings
    # ------------------------------------------------------------------

    humoid_gemma4_mode: Gemma4Mode = "auto"
    humoid_gemma4_thinking: Gemma4ThinkingMode = "low"

    humoid_gemma4_native_fallback: bool = True
    humoid_gemma4_strip_completed_thoughts: bool = True
    # Start the managed llama.cpp server and select the local Gemma provider
    # during app startup and before prompts when necessary.
    humoid_gemma_autostart: bool = False

    # ------------------------------------------------------------------
    # Memory routing
    #
    # auto:
    #   1. remote Weaviate
    #   2. embedded Weaviate
    #   3. SQLite
    #
    # embedded-weaviate:
    #   launch Weaviate directly from Python without Docker
    # ------------------------------------------------------------------

    humoid_memory_backend: MemoryBackendName = "embedded-weaviate"

    humoid_memory_db: Path = Path("./.humoid/memory.sqlite3")

    humoid_memory_search_limit: int = 8
    humoid_memory_context_limit: int = 6
    humoid_memory_temporal_radius: int = 3
    humoid_memory_packet_max_chars: int = 12000

    # ------------------------------------------------------------------
    # Remote Weaviate settings
    # ------------------------------------------------------------------

    weaviate_http_host: str = "127.0.0.1"
    weaviate_http_port: int = 8080

    weaviate_grpc_host: str = "127.0.0.1"
    weaviate_grpc_port: int = 50051

    weaviate_secure: bool = False
    weaviate_api_key: str = ""

    # ------------------------------------------------------------------
    # Embedded Weaviate settings
    #
    # No Docker is required.
    # The Python client downloads and starts the native binary.
    # ------------------------------------------------------------------

    weaviate_embedded_version: str = "1.37.0"

    weaviate_embedded_data_path: Path = Path(
        "./.humoid/weaviate-data"
    )

    weaviate_embedded_binary_path: Path = Path(
        "./.humoid/weaviate-bin"
    )

    weaviate_embedded_log_level: str = "error"
    weaviate_embedded_disable_telemetry: bool = True
    weaviate_embedded_http_port: int = 8079
    weaviate_embedded_grpc_port: int = 50050

    # ------------------------------------------------------------------
    # Weaviate collection
    # ------------------------------------------------------------------

    weaviate_collection: str = "HumoidMemoryV1"

    # ------------------------------------------------------------------
    # Memory schema defaults
    # ------------------------------------------------------------------

    humoid_default_memory_tier: str = "episodic"
    humoid_default_memory_channel: str = "agent"
    humoid_default_validation_status: str = "unverified"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator(
        "digitalocean_base_url",
        "meta_base_url",
        "openai_base_url",
        "llamacpp_base_url",
        "litert_base_url",
        mode="before",
    )
    @classmethod
    def normalize_base_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        normalized = value.strip().rstrip("/")

        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]

        return normalized

    @field_validator("humoid_workspace", mode="after")
    @classmethod
    def normalize_workspace(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator(
        "humoid_memory_db",
        "weaviate_embedded_data_path",
        "weaviate_embedded_binary_path",
        mode="after",
    )
    @classmethod
    def normalize_storage_path(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("humoid_max_tool_rounds")
    @classmethod
    def validate_tool_rounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "HUMOID_MAX_TOOL_ROUNDS must be at least 1"
            )

        if value > 100:
            raise ValueError(
                "HUMOID_MAX_TOOL_ROUNDS must not exceed 100"
            )

        return value

    @field_validator("humoid_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError(
                "HUMOID_TEMPERATURE must be between 0 and 2"
            )

        return value

    @field_validator(
        "humoid_tool_argument_max_bytes",
        "humoid_tool_result_max_bytes",
    )
    @classmethod
    def validate_byte_limit(cls, value: int) -> int:
        if value < 1024:
            raise ValueError(
                "Tool byte limits must be at least 1024 bytes"
            )

        return value

    @field_validator(
        "humoid_context_limit",
        "humoid_gemma_context_limit",
        "humoid_glm_context_limit",
    )
    @classmethod
    def validate_context_limit(cls, value: int) -> int:
        if value < 1024:
            raise ValueError("Context windows must be at least 1024 tokens")
        return value

    def context_limit(self, name: ProviderName | None = None) -> int:
        """Return the context window configured for the selected model family."""
        cfg = self.provider(name)
        model = cfg.model.lower()
        if cfg.name in {"llamacpp", "litert"} and "gemma" in model:
            return self.humoid_gemma_context_limit
        if cfg.name == "digitalocean" and "glm" in model:
            return self.humoid_glm_context_limit
        return self.humoid_context_limit

    # ------------------------------------------------------------------
    # Provider lookup
    # ------------------------------------------------------------------

    def provider(
        self,
        name: ProviderName | None = None,
    ) -> ProviderConfig:
        provider_name = name or self.humoid_provider

        if provider_name == "digitalocean":
            return ProviderConfig(
                name="digitalocean",
                api_key=self.digitalocean_api_key,
                base_url=self.digitalocean_base_url,
                model=self.digitalocean_model,
                timeout_seconds=self.digitalocean_timeout_seconds,
                max_retries=self.digitalocean_max_retries,
            )

        if provider_name == "meta":
            extra_body: dict[str, object] = {}

            if self.meta_enable_search_grounding:
                extra_body = {
                    "search_grounding": {
                        "enabled": True,
                    }
                }

            return ProviderConfig(
                name="meta",
                api_key=self.meta_api_key,
                base_url=self.meta_base_url,
                model=self.meta_model,
                timeout_seconds=self.meta_timeout_seconds,
                max_retries=self.meta_max_retries,
                extra_body=extra_body,
            )

        if provider_name == "openai":
            return ProviderConfig(
                name="openai",
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                model=self.openai_model,
                timeout_seconds=self.openai_timeout_seconds,
                max_retries=self.openai_max_retries,
            )

        if provider_name == "llamacpp":
            return ProviderConfig(
                name="llamacpp",
                api_key=self.llamacpp_api_key,
                base_url=self.llamacpp_base_url,
                model=self.llamacpp_model,
                timeout_seconds=self.llamacpp_timeout_seconds,
                max_retries=self.llamacpp_max_retries,
            )

        if provider_name == "litert":
            return ProviderConfig(
                name="litert",
                api_key=self.litert_api_key,
                base_url=self.litert_base_url,
                model=self.litert_model,
                timeout_seconds=self.litert_timeout_seconds,
                max_retries=self.litert_max_retries,
            )

        raise ValueError(
            f"Unsupported provider: {provider_name}"
        )

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def ensure_directories(self) -> None:
        self.humoid_workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.humoid_memory_db.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.weaviate_embedded_data_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.weaviate_embedded_binary_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    def gemma4_active(
        self,
        provider_name: ProviderName | None = None,
        model_name: str | None = None,
    ) -> bool:
        if self.humoid_gemma4_mode == "off":
            return False

        if self.humoid_gemma4_mode == "native":
            return True

        provider = self.provider(provider_name)

        model = (
            model_name
            or provider.model
            or ""
        ).lower()

        return (
            "gemma-4" in model
            or "gemma4" in model
        )

    def selected_tool_protocol(
        self,
        provider_name: ProviderName | None = None,
        model_name: str | None = None,
    ) -> ToolProtocolName:
        if self.humoid_tool_protocol != "auto":
            return self.humoid_tool_protocol

        provider = self.provider(provider_name)

        model = (
            model_name
            or provider.model
            or ""
        ).lower()

        identity = f"{provider.name} {model}"

        if self.gemma4_active(
            provider_name=provider.name,
            model_name=model,
        ):
            return "gemma4"

        if provider.name in {
            "openai",
            "digitalocean",
            "meta",
        }:
            return "openai"

        if any(
            token in identity
            for token in (
                "glm",
                "qwen",
                "hermes",
                "functionary",
                "deepseek",
            )
        ):
            return "tagged-json"

        if provider.name in {
            "llamacpp",
            "litert",
        }:
            return "humoid-v1"

        return "openai"

    def validate_provider_configuration(
        self,
        name: ProviderName | None = None,
    ) -> None:
        provider = self.provider(name)

        if not provider.base_url:
            raise ValueError(
                f"{provider.name} base URL is not configured"
            )

        if not provider.model:
            raise ValueError(
                f"{provider.name} model is not configured"
            )

        if provider.name not in {
            "llamacpp",
            "litert",
        } and not provider.api_key:
            raise ValueError(
                f"{provider.name} API key is not configured"
            )
