# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Per-layer KV cache CPU offloading via LMCache connector infrastructure.

This module implements a KV cache offloading strategy where the GPU only
holds a small buffer (default: 2 layers worth of KV cache), while the full
KV cache for all layers is stored in CPU pinned memory.  Before each layer's
attention computation the corresponding KV data is copied from CPU to the
GPU buffer, and after attention completes the data is copied back to CPU.
Double-buffering overlaps data transfer with computation.

Usage:
    Configure via ``kv_connector_extra_config``:
        ``full_offload: true``
        ``num_gpu_buffer_layers: 2``  (optional, default 2)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = init_logger(__name__)


class FullOffloadEngine:
    """Manages per-layer KV cache CPU offloading with double-buffering.

    This engine is used by ``LMCacheConnectorV1`` when ``full_offload`` mode
    is enabled.  It does NOT depend on the LMCache library itself -- it only
    reuses the LMCache connector hooks (``start_load_kv``,
    ``wait_for_layer_load``, ``save_kv_layer``, ``wait_for_save``).
    """

    def __init__(
        self,
        num_gpu_buffer_layers: int = 2,
    ):
        self.num_gpu_buffer_layers = num_gpu_buffer_layers

        # Populated by register_kv_caches / register_cpu_kv_caches
        self.gpu_kv_caches: dict[str, torch.Tensor] = {}
        self.cpu_kv_caches: dict[str, torch.Tensor] = {}

        # Ordered list of layer names (set during register_kv_caches)
        self.layer_names_ordered: list[str] = []
        self.layer_name_to_idx: dict[str, int] = {}
        self.num_layers: int = 0

        # CUDA streams & events for async transfer (created lazily)
        self._streams: list[torch.cuda.Stream] = []
        self._load_events: list[torch.cuda.Event] = []
        self._save_events: list[torch.cuda.Event] = []
        self._save_events_recorded: list[bool] = []
        self._initialized: bool = False

        # Per-step state
        self.current_layer: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def register_kv_caches(
        self,
        gpu_kv_caches: dict[str, torch.Tensor],
        cpu_kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register the GPU buffer and CPU backing store tensors."""
        self.gpu_kv_caches = gpu_kv_caches
        self.cpu_kv_caches = cpu_kv_caches

        # Determine layer ordering from dict insertion order
        self.layer_names_ordered = list(gpu_kv_caches.keys())
        self.layer_name_to_idx = {
            name: idx for idx, name in enumerate(self.layer_names_ordered)
        }
        self.num_layers = len(self.layer_names_ordered)

        logger.info(
            "FullOffloadEngine: registered %d layers, "
            "%d GPU buffer slots",
            self.num_layers,
            self.num_gpu_buffer_layers,
        )

    def _lazy_init_streams(self) -> None:
        """Create CUDA streams and events on first use."""
        if self._initialized:
            return
        for _ in range(self.num_gpu_buffer_layers):
            self._streams.append(torch.cuda.Stream())
            self._load_events.append(torch.cuda.Event())
            self._save_events.append(torch.cuda.Event())
            self._save_events_recorded.append(False)
        self._initialized = True

    # ------------------------------------------------------------------
    # Per-step lifecycle (called from the connector)
    # ------------------------------------------------------------------

    def start_load_kv(
        self,
        forward_context: ForwardContext,
        **kwargs: Any,
    ) -> None:
        """Initiate async load of layer 0's KV from CPU to GPU buffer.

        Called once per forward step, before the layer loop.
        """
        self._lazy_init_streams()
        self.current_layer = 0

        if self.num_layers == 0:
            return

        # Prefetch layer 0 into its GPU buffer slot
        self._async_load_layer(0)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for *layer_name* is loaded into the GPU buffer.

        Also kicks off async prefetch of the next layer.
        """
        layer_idx = self.layer_name_to_idx[layer_name]
        slot = layer_idx % self.num_gpu_buffer_layers

        # Wait for the load of this layer to complete on the compute stream
        torch.cuda.current_stream().wait_event(self._load_events[slot])

        # Prefetch the next layer into the other buffer slot
        next_idx = layer_idx + 1
        if next_idx < self.num_layers:
            # First, ensure any pending save on the next slot is done
            # before we overwrite it with new data
            next_slot = next_idx % self.num_gpu_buffer_layers
            if self._save_events_recorded[next_slot]:
                self._streams[next_slot].wait_event(
                    self._save_events[next_slot]
                )
            self._async_load_layer(next_idx)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Initiate async save of the current layer's KV from GPU buffer to CPU."""
        layer_idx = self.layer_name_to_idx[layer_name]
        slot = layer_idx % self.num_gpu_buffer_layers
        stream = self._streams[slot]

        cpu_tensor = self.cpu_kv_caches[layer_name]
        gpu_tensor = self.gpu_kv_caches[layer_name]

        # Wait for compute to finish before copying back
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            cpu_tensor.copy_(gpu_tensor, non_blocking=True)
            self._save_events[slot].record(stream)
            self._save_events_recorded[slot] = True

        self.current_layer += 1

    def wait_for_save(self) -> None:
        """Block until all pending GPU->CPU saves are complete."""
        for slot in range(self.num_gpu_buffer_layers):
            if self._save_events_recorded[slot]:
                self._save_events[slot].synchronize()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _async_load_layer(self, layer_idx: int) -> None:
        """Async copy layer's KV data from CPU to its GPU buffer slot."""
        layer_name = self.layer_names_ordered[layer_idx]
        slot = layer_idx % self.num_gpu_buffer_layers
        stream = self._streams[slot]

        cpu_tensor = self.cpu_kv_caches[layer_name]
        gpu_tensor = self.gpu_kv_caches[layer_name]

        with torch.cuda.stream(stream):
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            self._load_events[slot].record(stream)

    def get_gpu_kv_cache_for_layer(self, layer_name: str) -> torch.Tensor:
        """Return the GPU buffer tensor for the given layer."""
        return self.gpu_kv_caches[layer_name]
