import torch


class WhisperTimestampLogitsProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.no_ts_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self.ts_begin = tokenizer.convert_tokens_to_ids("<|0.00|>")
        if self.ts_begin is None or self.ts_begin < 0:
            raise RuntimeError("Tokenizer has no timestamp token")

    def __call__(self, logits: torch.Tensor, seqs: list, is_prefill: bool):
        # if is_prefill:
        #     return logits
        # Should use mask for performance, but just POC now.
        out = logits.clone()
        for i, seq in enumerate(seqs):
            if not getattr(seq, "return_timestamps", False):
                continue
            out[i, self.no_ts_id] = -float("inf")  # ban no timestamp
            if seq.last_timestamp_id is not None:
                out[i, self.ts_begin:seq.last_timestamp_id] = -float("inf")
        return out
