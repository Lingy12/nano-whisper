import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from nanowhisper.config import Config
from nanowhisper.sampling_params import SamplingParams
from nanowhisper.engine.sequence import Sequence
from nanowhisper.engine.scheduler import Scheduler
from nanowhisper.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        try:
            config.no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        except Exception:
            config.no_timestamps_token_id = getattr(self.tokenizer, "no_timestamps_token_id", 50363)
        config.timestamp_begin = config.no_timestamps_token_id + 1
        config.time_precision = getattr(self.tokenizer, "time_precision", 0.02)
        self.scheduler = Scheduler(config)
        self._seq_params = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: dict[str, list[int] | dict[str, torch.Tensor]], sampling_params: SamplingParams):
        if isinstance(prompt["prompt"], str):
            if sampling_params.return_timestamps:
                prompt["prompt"] = prompt["prompt"].replace("<|notimestamps|>", "")
            prompt["prompt"] = self.tokenizer.encode(prompt["prompt"], add_special_tokens = False)
        else:
            if sampling_params.return_timestamps:
                no_timestamps_token_id = getattr(self.tokenizer, "no_timestamps_token_id", None)
                if no_timestamps_token_id is None:
                    try:
                        no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
                    except Exception:
                        no_timestamps_token_id = None
                if no_timestamps_token_id is not None:
                    prompt["prompt"] = [t for t in prompt["prompt"] if t != no_timestamps_token_id]
        seq = Sequence(prompt.get("prompt", None), sampling_params, input_tensors = prompt.get("multi_modal_data", None))
        self._seq_params[seq.seq_id] = sampling_params
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def _get_timestamp_begin(self):
        no_timestamps_token_id = getattr(self.tokenizer, "no_timestamps_token_id", None)
        if no_timestamps_token_id is None:
            try:
                no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            except Exception:
                no_timestamps_token_id = getattr(self.tokenizer, "no_timestamps_token_id", 50363)
        return no_timestamps_token_id + 1

    def _build_segments(self, token_ids: list[int]):
        timestamp_begin = self._get_timestamp_begin()
        time_precision = getattr(self.tokenizer, "time_precision", 0.02)
        segments = []
        current_start = None
        current_tokens = []
        for token_id in token_ids:
            if token_id >= timestamp_begin:
                t = (token_id - timestamp_begin) * time_precision
                if current_start is None:
                    current_start = t
                else:
                    text = self.tokenizer.decode(current_tokens, skip_special_tokens=True)
                    segments.append({"start": float(current_start), "end": float(t), "text": text})
                    current_start = t
                    current_tokens = []
            else:
                current_tokens.append(token_id)
        return segments

    def generate(
        self,
        prompts: list[dict[str, list[int] | dict[str, torch.Tensor]]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        ordered_seq_ids = sorted(outputs)
        outputs = [outputs[seq_id] for seq_id in ordered_seq_ids]
        decoded = []
        for seq_id, token_ids in zip(ordered_seq_ids, outputs):
            result = {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            sp = self._seq_params.pop(seq_id, None)
            if sp and sp.return_timestamps:
                segments = self._build_segments(token_ids)
                result["segments"] = segments
            decoded.append(result)
        if use_tqdm:
            pbar.close()
        return decoded

