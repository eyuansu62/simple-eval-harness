import os
import copy
import json
import logging
import hashlib
import itertools
import asyncio
from tqdm import tqdm
from functools import cached_property
from importlib.util import find_spec
from sqlitedict import SqliteDict


try:
    import requests
    from aiohttp import ClientSession, TCPConnector, ClientTimeout, ClientResponseError
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
    from tqdm.asyncio import tqdm_asyncio
except ModuleNotFoundError:
    pass

import pdb

# utility class to keep track of json encoded chats
class JsonChatStr:
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
eval_logger = logging.getLogger("lm-eval")


### SQLite-based caching of LM responses
def hash_args(attr, req, conversation_id=None):
    """Generate hash for both single-turn and multi-turn requests"""
    if conversation_id is not None:
        # Multi-turn case
        dat = json.dumps([attr] + req + [conversation_id])
    else:
        # Single-turn case
        dat = json.dumps([attr] + req)
    return hashlib.sha256(dat.encode("utf-8")).hexdigest()


class CacheHook:
    def __init__(self, cachinglm) -> None:
        if cachinglm is None:
            self.cache = None
            return
        self.cache = cachinglm.cache

    def add_partial(self, attr, req, res, conversation_id=None) -> None:
        if self.cache is None:
            return
        hsh = hash_args(attr, req, conversation_id)
        self.cache[hsh] = res

class CachingLM:
    def __init__(self, lm, cache_db) -> None:
        """LM wrapper that returns cached results if they exist, and uses the underlying LM if not.

        :param lm: LM
            Underlying LM
        :param cache_db: str
            Path to cache db
        """
        self.lm = lm
        self.cache_db = cache_db
        if os.path.dirname(cache_db):
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
        self.cache = SqliteDict(cache_db, autocommit=True)

        # add hook to lm
        lm.set_cache_hook(self.get_cache_hook())

    def __getattr__(self, attr: str):
        eval_logger.info(f"Wrapping '{attr}' method of LM with caching...")

        def fn(requests, gen_kwargs):
            res = []
            remaining_reqs = []
            
            eval_logger.info(f"Loading '{attr}' responses from cache where possible...")
            for req in tqdm(requests, desc="Checking cached requests"):
                if isinstance(req, tuple):
                    # Multi-turn case
                    conversation_id, req_content = req
                    hsh = hash_args(attr, req_content, conversation_id)
                else:
                    # Single-turn case
                    hsh = hash_args(attr, req)
                    conversation_id, req_content = None, req

                if hsh in self.cache:
                    ob = self.cache[hsh]
                    assert ob is not None
                    res.append(ob)
                else:
                    res.append(None)
                    remaining_reqs.append((conversation_id, req_content) if conversation_id else req_content)

            eval_logger.info(
                f"Cached requests: {len(requests) - len(remaining_reqs)}, Requests remaining: {len(remaining_reqs)}"
            )

            if remaining_reqs:
                rem_res = getattr(self.lm, attr)(remaining_reqs, gen_kwargs=gen_kwargs)
            else:
                rem_res = []

            # Integrate new results
            resptr = 0
            for req, r in zip(remaining_reqs, rem_res):
                while res[resptr] is not None:
                    resptr += 1
                res[resptr] = r
                
                if isinstance(req, tuple):
                    # Multi-turn case
                    conversation_id, req_content = req
                    hsh = hash_args(attr, req_content, conversation_id)
                else:
                    # Single-turn case
                    hsh = hash_args(attr, req)
                self.cache[hsh] = r

            self.cache.commit()
            return res

        return fn

    def get_cache_hook(self):
        return CacheHook(self)
    
class TemplateAPI:
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        max_retries: int = 1000,
        num_concurrent: int = 1,
        max_gen_toks: int = 6144,
        seed: int = 1234,
        api_key: str = "xxx",
    ) -> None:
        missing_packages = [
            pkg
            for pkg in ["aiohttp", "tqdm", "tenacity", "requests"]
            if find_spec(pkg) is None
        ]
        if missing_packages:
            raise ModuleNotFoundError(
                f"Attempted to use an API model, but the required packages {missing_packages} are not installed. "
                'Please install these via `pip install lm-eval[api]` or `pip install -e ."[api]"`'
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._max_gen_toks = int(max_gen_toks)
        self._seed = int(seed)
        self.max_retries = int(max_retries)
        self._concurrent = int(num_concurrent)

    def set_cache_hook(self, cache_hook) -> None:
        self.cache_hook = cache_hook

    def _create_payload(
        self,
        messages: list,
        gen_kwargs: dict,
    ) -> dict:
        """
        Create the payload to send to the language model API.
        """
        gen_kwargs = gen_kwargs or {}
        gen_kwargs = copy.deepcopy(gen_kwargs)

        # Pop known generation arguments or use defaults
        max_tokens = gen_kwargs.pop("max_tokens", gen_kwargs.pop("max_gen_toks", self._max_gen_toks))
        temperature = gen_kwargs.pop("temperature", 0.0)
        stop = gen_kwargs.pop("until", ["<|endoftext|>"])
        seed = gen_kwargs.pop("seed", self._seed)

        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        if isinstance(messages, str):
            key = "prompt"
        else:
            key = "messages"

        payload = {
            key: messages,
            "model": self.model,
            "seed": seed,
            **gen_kwargs,
        }
        # print(payload)

       
        if "o1" in (self.model or "").lower():
            payload.update({
                "max_completion_tokens": 20000
            })
        elif self.model.startswith("ep-20"):
            payload.update({
                "max_tokens": 12288
            })
        elif "claude-3-7-sonnet-20250219#thinking" in (self.model or "").lower():
            payload.pop("seed")
        else:
            if not gen_kwargs.pop("seed", None): payload.pop("seed")
            payload.update({
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop[:4],
            })
        return payload


    def parse_generations(self, outputs: list, **kwargs):
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            for choices in out["choices"]:
                if "text" in choices:
                    res.append(choices['text'])
                else:
                    res.append(choices["message"]["content"])

                # if "reasoning_content" in choices['message']:
                #     # print(choices['message']['reasoning_content'])
                #     res[-1] = "<think>" + choices['message']['reasoning_content'] + "</think>" + res[-1]
        return res

    @cached_property
    def header(self) -> dict:
        """Override this property to return the headers for the API request."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _handle_inappropriate_content(self, text: str):
        """
        If the server response indicates inappropriate content, return a filtered response.
        """
        
        if any(
            keyword in text
            for keyword in ["inappropriate content", "high risk", "security reason", "敏感", "BadRequestError"]
        ):
            return {"choices": [{"message": {"content": "content filter"}}]}
        return {}

    def model_call(
        self,
        messages: list,
        gen_kwargs: dict,
    ):
        """
        Synchronous API call.
        """
        payload = self._create_payload(messages, gen_kwargs=gen_kwargs)

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.header,
            )
            # print(response.text)
            if not response.ok:
                eval_logger.warning(f"API request failed: {response.text}")
                filtered = self._handle_inappropriate_content(str(response.text))
                if filtered:
                    return filtered

            response.raise_for_status()
            return response.json()

        except RetryError:
            eval_logger.error("API request failed after multiple retries. Check API status.")
            return None

    async def amodel_call(
        self,
        session: ClientSession,
        message: list,
        gen_kwargs: dict,
        cache_keys: tuple,
    ):
        """
        Asynchronous API call.
        """
        payload = self._create_payload(message, gen_kwargs=gen_kwargs)
        try:
            async with session.post(
                self.base_url,
                json=payload,
                headers=self.header,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(f"API request failed: {error_text}")
                    filtered = self._handle_inappropriate_content(str(error_text))
                    outputs = filtered if filtered else None
                    if not outputs:
                        response.raise_for_status()
                else:
                    outputs = await response.json()

            if outputs is None:
                return None
            
            answers = self.parse_generations(outputs=outputs)
            if cache_keys:
                req, conversation_id = cache_keys
                self.cache_hook.add_partial("generate_until", req, answers, conversation_id)
            return answers

        except RetryError:
            eval_logger.error("API request failed after multiple retries. Check API status.")
            return None


    async def get_batched_requests(
        self,
        requests: list,
        gen_kwargs: dict,
    ):
        conn = TCPConnector(limit=self._concurrent)
        async with ClientSession(connector=conn, timeout=ClientTimeout(total=6 * 60 * 60)) as session:
            retry_ = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
            )(self.amodel_call)

            if isinstance(requests[0], tuple):
                tasks = [
                    asyncio.create_task(
                        retry_(
                            session=session,
                            message=message,
                            gen_kwargs=gen_kwargs,
                            cache_keys=(message, conversation_id),
                        )
                    )
                    for conversation_id, message in requests
                ]

            else:
                tasks = [
                    asyncio.create_task(
                        retry_(
                            session=session,
                            message=message,
                            gen_kwargs=gen_kwargs,
                            cache_keys=None,
                        )
                    )
                    for message in requests
                ]
            return await tqdm_asyncio.gather(*tasks, desc=f"Requesting API: {self.model}")
        
    def generate_until(
        self,
        requests: list,
        gen_kwargs: dict = None
    ):
        res = []
        if self._concurrent == 1:
            pbar = tqdm(desc=f"Requesting API: {self.model}", total=len(requests))
            for req in requests:
                # print(req)
                use_cache = False
                if isinstance(req, tuple):
                    conversation_id, req = req
                    use_cache = True
                outputs = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=0.5, min=1, max=10),
                    reraise=True,
                )(self.model_call)(req, gen_kwargs)
                if outputs:
                    generated_text = self.parse_generations(outputs)
                    res.extend(generated_text)
                    if use_cache:
                        self.cache_hook.add_partial(
                            "generate_until",
                            req,
                            generated_text,
                            conversation_id
                        )
                pbar.update(1)
        else:
            results = itertools.chain.from_iterable(
                asyncio.run(
                    self.get_batched_requests(
                        requests,
                        gen_kwargs=copy.deepcopy(gen_kwargs)
                    )
                )
            )
            res.extend(results)

        return res
