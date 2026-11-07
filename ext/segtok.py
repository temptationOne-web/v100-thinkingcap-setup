#!/usr/bin/env python3
"""segtok - incremental (segment-cached) prompt tokenizer for the brain's API server.  STAGED 2026-10-30, NOT LIVE.

Why: every JARVIS call re-tokenizes the whole rendered chat (vLLM API server, AsyncMicrobatchTokenizer, one thread):
~0.29-0.32 s per 100k tokens on this box (measured offline on the owner's real chats, see segtok_test.py), i.e.
~0.45 s at 148k and ~0.7 s at 227k, paid on EVERY tool-loop step, in series with priority-client's requests on the same
tokenizer thread. Between two calls of one chat only the last few messages are new.

How (exact, not approximate): the chat template separates messages with the special token <|im_start|>. HF fast
tokenizers split the text on added/special tokens BEFORE any merging, so the text between two <|im_start|> markers is
always tokenized independently of its neighbours. We cut the rendered prompt at every <|im_start|>, tokenize each piece
once, cache its ids by a 128-bit hash of the piece, and concatenate. A repeated piece costs a dict lookup.
segtok_test.py checks ids == full encode on real chats before this may be used.

Integration (needs a brain restart = owner window + gates; not done): wrap the tokenizer the API server uses, e.g. a
vLLM general plugin (entry point vllm.general_plugins) that patches the renderer's tokenizer encode for single
string prompts with add_special_tokens=False. Kill switch: env SEGTOK=0.
"""
import hashlib
import os
from array import array
from collections import OrderedDict

BOUNDARY = "<|im_start|>"


class SegmentCachedEncoder:
    def __init__(self, tokenizer, boundary=BOUNDARY, max_tokens=3_000_000):
        self.tok = tokenizer
        self.boundary = boundary
        self.max_tokens = max_tokens      # ids kept as uint32 arrays: 3M tokens = ~12 MB RAM
        self.size = 0
        self.cache = OrderedDict()
        self.hits = self.misses = 0
        self.enabled = os.environ.get("SEGTOK", "1") != "0"

    def _pieces(self, text):
        b = self.boundary
        start = 0
        while True:
            j = text.find(b, start + 1 if start else 1)
            if j < 0:
                yield text[start:]
                return
            yield text[start:j]
            start = j

    def encode(self, text):
        """ids for `text` with add_special_tokens=False (the chat template already contains every special token)."""
        if not self.enabled or self.boundary not in text:
            return self.tok.encode(text, add_special_tokens=False)
        out = []
        c = self.cache
        for piece in self._pieces(text):
            if not piece:
                continue
            k = hashlib.blake2b(piece.encode("utf-8", "surrogatepass"), digest_size=16).digest()
            ids = c.get(k)
            if ids is None:
                ids = array("I", self.tok.encode(piece, add_special_tokens=False))
                c[k] = ids
                self.size += len(ids)
                self.misses += 1
                while self.size > self.max_tokens and len(c) > 1:
                    self.size -= len(c.popitem(last=False)[1])
            else:
                c.move_to_end(k)
                self.hits += 1
            out.extend(ids)
        return out
