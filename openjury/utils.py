import time
import asyncio
import os
import warnings
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
import pandas as pd
from tqdm.asyncio import tqdm
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

data_root = Path(
    os.environ.get("OPENJURY_DATA", Path("~/openjury-eval-data/").expanduser())
).expanduser()


def set_langchain_cache():
    set_llm_cache(SQLiteCache(database_path=str(data_root / ".langchain.db")))


def download_hf(name: str, local_path: Path):
    local_path.mkdir(exist_ok=True, parents=True)
    # downloads the model from huggingface into `local_path` folder
    snapshot_download(
        repo_id="geoalgo/llmjudge",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **pandas_kwargs)


def do_inference(chat_model, inputs, use_tqdm: bool = True):
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    if use_tqdm:
        # perform inference asynchronously to be able to update tqdm, chat_model.batch does not work as it blocks until
        # all requests are received
        async def process_with_real_progress(chat_model, inputs):
            async def process_single(input_item):
                return await chat_model.ainvoke(input_item, **invoke_kwargs)

            # Create all tasks
            tasks = [asyncio.create_task(process_single(inp)) for inp in inputs]
            results = []

            # Track progress as tasks complete
            with tqdm(total=len(inputs)) as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    pbar.update(1)

            return results

        res = asyncio.run(
            process_with_real_progress(chat_model=chat_model, inputs=inputs)
        )
    else:
        res = chat_model.batch(inputs=inputs, **invoke_kwargs)

    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
    # is it because of using Chat and barebones models?
    # when using OpenAI, the output is AIMessage not a string...
    res = [x.content if hasattr(x, "content") else x for x in res]
    return res


class DummyModel:
    def __init__(self, name: str):
        self.name = name
        self.message = "/".join(name.split("/")[1:])

    def batch(self, inputs, **invoke_kwargs) -> list[str]:
        return [self.message] * len(inputs)

    def invoke(self, input, **invoke_kwargs) -> str:
        return self.message

    async def ainvoke(self, input, **invoke_kwargs):
        return self.message


class ChatVLLM:
    """VLLM wrapper that auto-detects whether to use chat() or generate().

    Chat template handling:
        - If ``chat_template`` is explicitly provided, always uses ``llm.chat()``
          with that template (useful for models whose tokenizer lacks a template
          but you know the correct one).
        - If the tokenizer defines a chat template, uses ``llm.chat()`` and lets
          vLLM apply the tokenizer's template automatically.
        - If no chat template is found (typical for base/pretrained models),
          falls back to ``llm.generate()`` and emits a warning.  This avoids the
          ``ValueError`` raised by ``transformers >= v4.44`` which removed the
          default chat template.

    Truncation:
        Input text is truncated *before* it reaches this wrapper — see the
        ``--truncate_all_input_chars`` CLI flag, which caps the character length
        of instructions (before models A/B) and completions (before the judge).
        The ``--max_out_tokens_models`` / ``--max_out_tokens_judge`` flags control
        the ``max_tokens`` sampling parameter for generation length.
    """

    def __init__(self, model: str, max_tokens: int = 8192, chat_template: str | None = None, **vllm_kwargs):
        from vllm import LLM, SamplingParams

        self.model_path = model
        self.max_tokens = max_tokens
        self.llm = LLM(model=model, trust_remote_code=True, **vllm_kwargs)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
        )

        # Resolve chat template:
        # 1. Explicit override always wins → use chat() with that template
        # 2. If tokenizer has one, use it → use chat() (pass None to vLLM)
        # 3. No template found → fall back to generate() for base models
        if chat_template:
            self.chat_template = chat_template
            self._use_generate = False
        else:
            tokenizer = self.llm.get_tokenizer()
            if not getattr(tokenizer, "chat_template", None):
                warnings.warn(
                    f"Model '{model}' tokenizer does not define a chat template. "
                    f"Falling back to llm.generate() (no chat formatting). "
                    f"Override with --chat_template if this model needs one.",
                )
                self.chat_template = None
                self._use_generate = True
            else:
                self.chat_template = None  # let vLLM use the tokenizer's own
                self._use_generate = False

    def _to_messages(self, input_item) -> list[dict]:
        """Convert LangChain prompt input to OpenAI-style messages."""
        # Map LangChain message types to OpenAI roles
        role_map = {"human": "user", "ai": "assistant", "system": "system"}

        # Handle ChatPromptValue from LangChain
        if hasattr(input_item, "to_messages"):
            lc_messages = input_item.to_messages()
            return [
                {"role": role_map.get(msg.type, msg.type), "content": msg.content}
                for msg in lc_messages
            ]
        # Handle list of tuples like [("system", "..."), ("user", "...")]
        elif isinstance(input_item, list) and input_item and isinstance(input_item[0], tuple):
            return [
                {"role": role if role != "human" else "user", "content": content}
                for role, content in input_item
            ]
        # Handle already formatted messages
        elif isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
            return input_item
        # Handle plain string (wrap as user message)
        elif isinstance(input_item, str):
            return [{"role": "user", "content": input_item}]
        else:
            raise ValueError(f"Unsupported input type: {type(input_item)}")

    def _to_raw_text(self, input_item) -> str:
        """Extract raw text from an input item for use with llm.generate()."""
        if isinstance(input_item, str):
            return input_item
        # ChatPromptValue from LangChain
        if hasattr(input_item, "to_string"):
            return input_item.to_string()
        # List of dicts (messages) — concatenate contents
        if isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
            return "\n".join(msg["content"] for msg in input_item)
        raise ValueError(f"Cannot extract raw text from: {type(input_item)}")

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
        """Process a batch of inputs using vllm.LLM.chat() or llm.generate().

        Uses ``llm.chat()`` when a chat template is available (instruct models),
        and ``llm.generate()`` when no template is found (base models).
        """
        if self._use_generate:
            prompts = [self._to_raw_text(inp) for inp in inputs]
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            messages_batch = [self._to_messages(inp) for inp in inputs]
            outputs = self.llm.chat(
                messages_batch,
                self.sampling_params,
                add_generation_prompt=True,
                chat_template=self.chat_template,
            )
        return [out.outputs[0].text for out in outputs]

    def invoke(self, input_item, **invoke_kwargs) -> str:
        """Process a single input."""
        results = self.batch([input_item], **invoke_kwargs)
        return results[0]

    async def ainvoke(self, input_item, **invoke_kwargs):
        """Async version - runs sync version in executor for compatibility."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input_item, **invoke_kwargs))


def make_model(model: str, max_tokens: int | None = 8192, chat_template: str | None = None):
    """Instantiate a model wrapper from a provider/model-name string.

    Args:
        model: Format ``{Provider}/{model_path}``, e.g.
            ``VLLM/meta-llama/Llama-3.3-70B-Instruct``.
        max_tokens: Maximum tokens the model may generate.
        chat_template: Optional Jinja2 chat template override.  Only used by
            the VLLM provider; silently ignored for other providers.
    """
    model_provider = model.split("/")[0]

    if model_provider == "Dummy":
        return DummyModel(model)

    model_name = "/".join(model.split("/")[1:])
    print(f"Loading {model_provider}(model={model_name})")

    # Use our custom ChatVLLM wrapper which properly applies chat templates
    if model_provider == "VLLM":
        return ChatVLLM(
            model=model_name,
            max_tokens=max_tokens if max_tokens else 8192,
            chat_template=chat_template,
        )

    model_kwargs = {}
    if max_tokens is not None:
        model_kwargs["max_tokens"] = max_tokens

    if model_provider == "OpenRouter":
        # Special case we need to override API url and key
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            **model_kwargs,
        )
    else:
        model_classes = [
            LlamaCpp,
            ChatOpenAI,
        ]
        model_kwargs["model"] = model_name

        try:
            from langchain_together.llms import Together

            model_classes.append(Together)
        except ImportError as e:
            print(str(e))
        try:
            from langchain_openai.llms import OpenAI

            model_classes.append(OpenAI)
        except ImportError:
            print(str(e))
        model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
        assert (
            model_provider in model_cls_dict
        ), f"{model_provider} not available, choose among {list(model_cls_dict.keys())}"
        return model_cls_dict[model_provider](**model_kwargs)


def download_all():
    print(f"Downloading all dataset in {data_root}")
    for dataset in ["alpaca-eval", "arena-hard", "m-arena-hard"]:
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)

    snapshot_download(
        repo_id="geoalgo/multilingual-contexts-to-be-completed",
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )


class Timeblock:
    """Timer context manager"""

    def __init__(self, name: str | None = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        name = self.name if self.name else "block"
        msg = f"{name} took {self.duration} seconds"
        return msg


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
):
    f"""
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = data_root / "cache"
    cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        return pd.read_csv(cache_file)
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            df.to_csv(cache_file, index=False)
            return pd.read_csv(cache_file)


if __name__ == "__main__":
    download_all()
