#!/usr/bin/env python3
"""Apply the L1/L2 hit & serve attribution patch to a LMCache checkout.

Anchored string edits; every anchor is asserted unique before replacing.
Run from the repo root.
"""
import sys

EDITS = []  # (path, anchor, replacement)


def edit(path, anchor, replacement):
    EDITS.append((path, anchor, replacement))


# --------------------------------------------------------------------------
# 1. custom_types.py — wire field (same compat pattern as cache_salt)
# --------------------------------------------------------------------------
edit(
    "lmcache/v1/multiprocess/custom_types.py",
    '    cache_salt: str = ""\n',
    '    cache_salt: str = ""\n'
    "\n"
    "    # === GPU-native hit at lookup time (not part of cache identity) ===\n"
    "    # vLLM's local prefix-cache hit tokens at submission.  Lets serving\n"
    "    # metrics attribute the found prefix beyond the native hit to\n"
    "    # L1-resident vs L2-loaded chunks (see LookupModule).  Old payloads\n"
    "    # decode with the default 0 — same wire-compat pattern as\n"
    "    # ``cache_salt`` above.\n"
    "    native_hit_tokens: int = field(default=0, compare=False)\n",
)

edit(
    "lmcache/v1/multiprocess/custom_types.py",
    "            request_id=self.request_id,\n"
    "            cache_salt=self.cache_salt,\n"
    "        )",
    "            request_id=self.request_id,\n"
    "            cache_salt=self.cache_salt,\n"
    "            native_hit_tokens=self.native_hit_tokens,\n"
    "        )",
)

# --------------------------------------------------------------------------
# 2. vllm_multi_process_adapter.py — pass the native hit through
# --------------------------------------------------------------------------
edit(
    "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    "    def maybe_submit_lookup_request(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        token_ids: list[int],\n"
    '        cache_salt: str = "",\n'
    "    ):",
    "    def maybe_submit_lookup_request(\n"
    "        self,\n"
    "        request_id: str,\n"
    "        token_ids: list[int],\n"
    '        cache_salt: str = "",\n'
    "        native_hit_tokens: int = 0,\n"
    "    ):",
)

edit(
    "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    "                cache_salt values produce separate cache entries.\n",
    "                cache_salt values produce separate cache entries.\n"
    "            native_hit_tokens: vLLM's GPU-native prefix-cache hit token\n"
    "                count for this request, used server-side to attribute the\n"
    "                served (beyond-native) portion of the hit to L1-resident\n"
    "                vs L2-loaded chunks. 0 when unknown.\n",
)

edit(
    "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    "            request_id=request_id,\n"
    "            cache_salt=cache_salt,\n"
    "        ).no_worker_id_version()",
    "            request_id=request_id,\n"
    "            cache_salt=cache_salt,\n"
    "            native_hit_tokens=native_hit_tokens,\n"
    "        ).no_worker_id_version()",
)

edit(
    "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    "        request_id: str,\n"
    '        cache_salt: str = "",\n'
    "    ) -> IPCCacheServerKey:",
    "        request_id: str,\n"
    '        cache_salt: str = "",\n'
    "        native_hit_tokens: int = 0,\n"
    "    ) -> IPCCacheServerKey:",
)

edit(
    "lmcache/integration/vllm/vllm_multi_process_adapter.py",
    "        # NOTE: for the scheduler adapter, we don't have a worker id,\n"
    "        # so we set it to None in the key.\n"
    "        return IPCCacheServerKey(\n"
    "            model_name=self.model_name,\n"
    "            world_size=self.world_size,\n"
    "            worker_id=None,\n"
    "            token_ids=tuple(token_ids),\n"
    "            start=start,\n"
    "            end=end,\n"
    "            request_id=request_id,\n"
    "            cache_salt=cache_salt,\n"
    "        )",
    "        # NOTE: for the scheduler adapter, we don't have a worker id,\n"
    "        # so we set it to None in the key.\n"
    "        return IPCCacheServerKey(\n"
    "            model_name=self.model_name,\n"
    "            world_size=self.world_size,\n"
    "            worker_id=None,\n"
    "            token_ids=tuple(token_ids),\n"
    "            start=start,\n"
    "            end=end,\n"
    "            request_id=request_id,\n"
    "            cache_salt=cache_salt,\n"
    "            native_hit_tokens=native_hit_tokens,\n"
    "        )",
)

# --------------------------------------------------------------------------
# 3. lmcache_mp_connector.py — supply the native hit count (already in hand)
# --------------------------------------------------------------------------
edit(
    "lmcache/integration/vllm/lmcache_mp_connector.py",
    "        self.scheduler_adapter.maybe_submit_lookup_request(\n"
    "            request.request_id,\n"
    "            token_ids=list(request.all_token_ids),\n"
    "            cache_salt=tracker.cache_salt,\n"
    "        )",
    "        self.scheduler_adapter.maybe_submit_lookup_request(\n"
    "            request.request_id,\n"
    "            token_ids=list(request.all_token_ids),\n"
    "            cache_salt=tracker.cache_salt,\n"
    "            native_hit_tokens=num_computed_tokens,\n"
    "        )",
)

# --------------------------------------------------------------------------
# 4. modules/lookup.py — pure attribution helper + job field + emission
# --------------------------------------------------------------------------
edit(
    "lmcache/v1/multiprocess/modules/lookup.py",
    "@dataclass\nclass _PrefetchJob:",
    '''def _attribute_hit_chunks(
    found_leading_keys: int,
    l1_found_indices: tuple[int, ...],
    native_hit_tokens: int,
    chunk_size: int,
    keys_per_chunk: int,
) -> tuple[int, int, int, int]:
    """Split a lookup hit into (l1_hit, l2_hit, serve_l1, serve_l2) chunks.

    ``l1_hit``/``l2_hit`` split the whole found prefix by where each chunk
    was at submission time (presence attribution).  ``serve_l1``/``serve_l2``
    split only the portion beyond vLLM's GPU-native prefix hit — the chunks
    a retrieve will actually deliver (serve attribution).

    A chunk counts as L1-resident only when ALL of its keys were found in
    L1 at submission; partially-resident chunks attribute to L2, since
    completing them requires the L2 pipeline.

    Args:
        found_leading_keys: ``found.count_leading_ones()`` from the
            completed prefetch bitmap (contiguous found-key prefix).
        l1_found_indices: Original-key indices read-locked in L1 at
            submission time (``PrefetchHandle.l1_found_indices``).
        native_hit_tokens: vLLM's GPU-native prefix hit for the request;
            0 when unknown (older clients), which makes serve == hit.
        chunk_size: Tokens per chunk.
        keys_per_chunk: ``world_size * num_object_groups``.

    Returns:
        Tuple of chunk counts:
        ``(l1_hit_chunks, l2_hit_chunks, serve_l1_chunks, serve_l2_chunks)``.
    """
    if keys_per_chunk <= 0 or chunk_size <= 0:
        return 0, 0, 0, 0
    found_chunks = found_leading_keys // keys_per_chunk
    if found_chunks == 0:
        return 0, 0, 0, 0
    resident = set(l1_found_indices)
    l1_flags = [
        all((c * keys_per_chunk + k) in resident for k in range(keys_per_chunk))
        for c in range(found_chunks)
    ]
    l1_hit = sum(l1_flags)
    l2_hit = found_chunks - l1_hit
    native_chunks = min(native_hit_tokens // chunk_size, found_chunks)
    serve_l1 = sum(l1_flags[native_chunks:])
    serve_l2 = (found_chunks - native_chunks) - serve_l1
    return l1_hit, l2_hit, serve_l1, serve_l2


@dataclass
class _PrefetchJob:''',
)

edit(
    "lmcache/v1/multiprocess/modules/lookup.py",
    '    model_name: str = ""\n    cache_salt: str = ""\n',
    '    model_name: str = ""\n    cache_salt: str = ""\n'
    "    # vLLM's GPU-native prefix hit (tokens) at lookup submission; used\n"
    "    # for L1/L2 serve attribution at ``MP_LOOKUP_PREFETCH_END``.  0 when\n"
    "    # the client did not supply it.\n"
    "    native_hit_tokens: int = 0\n",
)

edit(
    "lmcache/v1/multiprocess/modules/lookup.py",
    "                num_object_groups=attn_desc.num_object_groups,\n"
    "                model_name=model_name,\n"
    "                cache_salt=key.cache_salt,\n"
    "            )\n"
    "        )",
    "                num_object_groups=attn_desc.num_object_groups,\n"
    "                model_name=model_name,\n"
    "                cache_salt=key.cache_salt,\n"
    "                native_hit_tokens=key.native_hit_tokens,\n"
    "            )\n"
    "        )",
)

edit(
    "lmcache/v1/multiprocess/modules/lookup.py",
    "        found_count = _get_prefix_hit_length(\n"
    "            found.count_leading_ones(), job.world_size, job.num_object_groups\n"
    "        )\n",
    "        found_count = _get_prefix_hit_length(\n"
    "            found.count_leading_ones(), job.world_size, job.num_object_groups\n"
    "        )\n"
    "\n"
    "        l1_hit_chunks, l2_hit_chunks, serve_l1_chunks, serve_l2_chunks = (\n"
    "            _attribute_hit_chunks(\n"
    "                found.count_leading_ones(),\n"
    "                job.handle.l1_found_indices,\n"
    "                job.native_hit_tokens,\n"
    "                self._ctx.chunk_size,\n"
    "                job.world_size * job.num_object_groups,\n"
    "            )\n"
    "        )\n",
)

edit(
    "lmcache/v1/multiprocess/modules/lookup.py",
    '                    "hit_tokens": found_count * self._ctx.chunk_size,\n',
    '                    "hit_tokens": found_count * self._ctx.chunk_size,\n'
    '                    "hit_tokens_l1_resident": l1_hit_chunks\n'
    "                    * self._ctx.chunk_size,\n"
    '                    "hit_tokens_l2_loaded": l2_hit_chunks * self._ctx.chunk_size,\n'
    '                    "serve_tokens_l1_resident": serve_l1_chunks\n'
    "                    * self._ctx.chunk_size,\n"
    '                    "serve_tokens_l2_loaded": serve_l2_chunks\n'
    "                    * self._ctx.chunk_size,\n",
)

# --------------------------------------------------------------------------
# 5. subscribers/metrics/lookup.py — four new counters
# --------------------------------------------------------------------------
edit(
    "lmcache/v1/mp_observability/subscribers/metrics/lookup.py",
    '        self._hit_tokens = meter.create_counter(\n'
    '            "lmcache_mp.lookup_hit",\n'
    '            description=(\n'
    '                "Total tokens found in L1+L2 during lookup (numerator of "\n'
    '                "the L1+L2 token-level hit rate). Counts the contiguous "\n'
    '                "prefix hit only."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n",
    '        self._hit_tokens = meter.create_counter(\n'
    '            "lmcache_mp.lookup_hit",\n'
    '            description=(\n'
    '                "Total tokens found in L1+L2 during lookup (numerator of "\n'
    '                "the L1+L2 token-level hit rate). Counts the contiguous "\n'
    '                "prefix hit only."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n"
    "        self._hit_l1_resident_tokens = meter.create_counter(\n"
    '            "lmcache_mp.lookup_hit_l1_resident",\n'
    "            description=(\n"
    '                "Of lookup_hit: tokens whose chunks were fully resident "\n'
    '                "in L1 at submission (presence attribution)."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n"
    "        self._hit_l2_loaded_tokens = meter.create_counter(\n"
    '            "lmcache_mp.lookup_hit_l2_loaded",\n'
    "            description=(\n"
    '                "Of lookup_hit: tokens whose chunks required the L2 "\n'
    '                "pipeline (presence attribution)."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n"
    "        self._serve_l1_resident_tokens = meter.create_counter(\n"
    '            "lmcache_mp.serve_l1_resident",\n'
    "            description=(\n"
    '                "Tokens beyond vLLM\'s GPU-native prefix hit served from "\n'
    '                "L1-resident chunks (serve attribution). Requires clients "\n'
    '                "that send native_hit_tokens; otherwise equals "\n'
    '                "lookup_hit_l1_resident."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n"
    "        self._serve_l2_loaded_tokens = meter.create_counter(\n"
    '            "lmcache_mp.serve_l2_loaded",\n'
    "            description=(\n"
    '                "Tokens beyond vLLM\'s GPU-native prefix hit that required "\n'
    '                "an L2 load (serve attribution). Requires clients that "\n'
    '                "send native_hit_tokens; otherwise equals "\n'
    '                "lookup_hit_l2_loaded."\n'
    "            ),\n"
    '            unit="tokens",\n'
    "        )\n",
)

edit(
    "lmcache/v1/mp_observability/subscribers/metrics/lookup.py",
    '        self._hit_tokens.add(event.metadata["hit_tokens"], attributes=attrs)\n',
    '        self._hit_tokens.add(event.metadata["hit_tokens"], attributes=attrs)\n'
    "        # .get(): events from emitters predating the L1/L2 attribution\n"
    "        # fields simply do not move the split counters.\n"
    "        self._hit_l1_resident_tokens.add(\n"
    '            event.metadata.get("hit_tokens_l1_resident", 0), attributes=attrs\n'
    "        )\n"
    "        self._hit_l2_loaded_tokens.add(\n"
    '            event.metadata.get("hit_tokens_l2_loaded", 0), attributes=attrs\n'
    "        )\n"
    "        self._serve_l1_resident_tokens.add(\n"
    '            event.metadata.get("serve_tokens_l1_resident", 0), attributes=attrs\n'
    "        )\n"
    "        self._serve_l2_loaded_tokens.add(\n"
    '            event.metadata.get("serve_tokens_l2_loaded", 0), attributes=attrs\n'
    "        )\n",
)


def main():
    failed = False
    for path, anchor, replacement in EDITS:
        with open(path) as f:
            src = f.read()
        n = src.count(anchor)
        if n != 1:
            print(f"ANCHOR NOT UNIQUE ({n}x) in {path}:\n---\n{anchor[:200]}\n---")
            failed = True
            continue
        with open(path, "w") as f:
            f.write(src.replace(anchor, replacement))
        print(f"patched {path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
