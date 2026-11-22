"""Build the detector of a KR Labs model from its model id."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError
from lettucedetect.detectors.generative import GenerativeConfig, GenerativeDetector
from lettucedetect.detectors.llm_client import LLMClient

__all__ = ["load_detector"]

# Every model id the library knows, with the detector class that runs it.
_REGISTRY: dict[str, type[BaseDetector]] = {
    "KRLabsOrg/lettucedect-base-modernbert-en-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-large-modernbert-en-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-v2-mmbert-base": EncoderDetector,
    "KRLabsOrg/lettucedect-mmbert-base-hu-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-de-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-pl-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-it-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-fr-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-es-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-210m-eurobert-cn-v1": EncoderDetector,
    "KRLabsOrg/lettucedect-610m-eurobert-de-v1": EncoderDetector,
    "KRLabsOrg/tinylettuce-ettin-17m-en": EncoderDetector,
    "KRLabsOrg/lettucedect-v2-qwen-2b": GenerativeDetector,
}


def _not_accepted(model_id: str, argument: str) -> ConfigurationError:
    return ConfigurationError(
        f"{_REGISTRY[model_id].__name__} ({model_id}) does not accept {argument!r}; "
        "remove the argument.",
        code="invalid_config",
        param=argument,
    )


def _pop_config(config_class: type[BaseModel], options: dict[str, Any]) -> Any:  # noqa: ANN401
    """Remove the config from ``options`` and return it.

    The config is the ``config`` option, or one built from the options that are fields of
    ``config_class``; ``None`` when neither is given.

    :raises ConfigurationError: (``invalid_config``) both forms are given, or a field is invalid.
    """
    fields = {name: options.pop(name) for name in config_class.model_fields if name in options}
    config = options.pop("config", None)
    if not fields:
        return config
    if config is not None:
        raise ConfigurationError(
            f"Pass the {config_class.__name__} options either in 'config' or as keyword "
            "arguments, not both.",
            code="invalid_config",
            param="config",
        )
    return config_class(**fields)


def load_detector(
    model_id: str,
    *,
    client: LLMClient | None = None,
    device: Any = None,  # noqa: ANN401
    model_version: str | None = None,
    **options: Any,  # noqa: ANN401
) -> BaseDetector:
    """Build the detector for a KR Labs model id.

    The id is looked up in the registry of the models the library knows; nothing is inferred
    from the name. For any other model build the detector class directly, for example
    ``EncoderDetector.from_pretrained`` for a local directory.

    An encoder id loads through :meth:`EncoderDetector.from_pretrained`, which may download.
    Its ``options`` are the fields of :class:`EncoderConfig` (or ``config``), ``taxonomy_head``,
    ``include_taxonomy`` and the keyword arguments of the Hugging Face loaders.

    A generative id builds a :class:`GenerativeDetector` on ``client``. Its ``options`` are the
    fields of :class:`GenerativeConfig` (or ``config``) and ``cache``.

    Example::

        detector = load_detector("KRLabsOrg/lettucedect-v2-qwen-2b", client=client)

    :param model_id: one of the registered model ids.
    :param client: the LLM client of a generative model; not accepted by an encoder id.
    :param device: device of an encoder model; not accepted by a generative id.
    :param model_version: version reported in ``usage``.
    :param options: options of the detector class of the id, see above.
    :returns: an :class:`EncoderDetector` or a :class:`GenerativeDetector`.
    :raises InvalidInputError: (``unknown_model_id``) ``model_id`` is not in the registry.
    :raises ConfigurationError: (``client_required``) a generative id without ``client``;
        (``invalid_config``) an argument the detector class of the id does not accept.
    """
    detector_class = _REGISTRY.get(model_id)
    if detector_class is None:
        raise InvalidInputError(
            f"Unknown model id; the known ids are: {', '.join(_REGISTRY)}. "
            "For any other model build the detector class directly.",
            code="unknown_model_id",
            param="model_id",
        )
    if detector_class is EncoderDetector:
        if client is not None:
            raise _not_accepted(model_id, "client")
        return EncoderDetector.from_pretrained(
            model_id,
            device=device,
            model_version=model_version,
            config=_pop_config(EncoderConfig, options),
            **options,
        )
    if client is None:
        raise ConfigurationError(
            f"{model_id} is a generative model and needs an LLM client; pass client=.",
            code="client_required",
            param="client",
        )
    if device is not None:
        raise _not_accepted(model_id, "device")
    config = _pop_config(GenerativeConfig, options)
    cache = options.pop("cache", None)
    if options:
        raise _not_accepted(model_id, next(iter(options)))
    return GenerativeDetector(
        client, model_id, config=config, cache=cache, model_version=model_version
    )
