# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


def _sm70_ct_nvfp4_tm_bridge_enabled() -> bool:
    """LOCAL PORT 2026-10-26: VLLM_SM70_CT_NVFP4_TM=1 on exact SM70 only."""
    import os

    if os.getenv("VLLM_SM70_CT_NVFP4_TM", "0") != "1":
        return False
    from vllm.platforms import current_platform

    return bool(current_platform.is_device_capability((7, 0)))


class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        use_a16: bool = False,
    ):
        super().__init__(moe)
        self.group_size = 16
        self.use_a16 = use_a16

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None if use_a16 else kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """
        if self.nvfp4_backend == NvFp4MoeBackend.SM70_SKINNY:
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.fused_moe.experts.skinny_sm70_moe import (
                nvfp4_skinny_scale_reason,
            )

            reason = nvfp4_skinny_scale_reason(
                1.0 / layer.w13_weight_global_scale,
                1.0 / layer.w2_weight_global_scale,
            )
            if reason is not None:
                if self.moe.moe_backend != "auto":
                    raise ValueError(reason)
                self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
                    self.moe,
                    kNvfp4Static,
                    None if self.use_a16 else kNvfp4Dynamic,
                    allow_skinny=False,
                )
                logger.info_once("Skinny MoE fallback: %s", reason)
                get_current_vllm_config().kernel_config.moe_kernel_selections[
                    f"Nvfp4SkinnySm70Experts:{self.moe.hidden_dim}:"
                    f"{self.moe.intermediate_size_per_partition}"
                ] = {
                    "enabled": False,
                    "reason": reason,
                    "scope": "loaded_scale_capability",
                }

        # NOTE(rob): wN_weight_packed -> wN_weight is because ModularKernelMethod
        # requires this naming convention. However, the name change breaks
        # reloading because the state dict no longer matches disk. Once we
        # remove MKM, we should revert this change to ensure compatibility.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # LOCAL PORT 2026-10-26 (why-slow): on exact SM70, hand the loaded NVFP4
        # experts to the ModelOpt SM70 TurboMind/QPN MoE route (the one the
        # engine qualifies for Qwen3.8 Flash-Next TP4, intermediate 160) instead
        # of the generic Marlin fallback. Opt-in: VLLM_SM70_CT_NVFP4_TM=1.
        if _sm70_ct_nvfp4_tm_bridge_enabled():
            self._sm70_bridge_to_modelopt_route(layer)
            return

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        # Shuffle weights into the NvFp4 kernel format.
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=(1.0 / w13_weight_global_scale),
            a13_scale=(1.0 / layer.w13_input_global_scale),
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=(1.0 / layer.w2_weight_global_scale),
            a2_scale=(1.0 / layer.w2_input_global_scale),
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w2_weight_scale_2 = w2_scale_2
        layer.w13_input_scale = a13_scale
        layer.w2_input_scale = a2_scale

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def _sm70_bridge_to_modelopt_route(self, layer: RoutedExperts) -> None:
        """LOCAL PORT: re-home compressed-tensors NVFP4 experts on the SM70
        ModelOpt route. Same e2m1 codes and fp8 block scales; only the global
        scale convention differs (compressed-tensors stores its reciprocal)."""
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )
        from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
            ModelOptNvFp4SM70MoEMethod,
        )

        w13_g = layer.w13_weight_global_scale.detach().float()  # [E, 2] (gate, up)
        w2_g = layer.w2_weight_global_scale.detach().float()  # [E]
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            (1.0 / w13_g).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            (1.0 / w2_g).contiguous(), requires_grad=False
        )
        # Weight-only (A16): the SM70 method never reads these, but deletes them.
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        for name in (
            "w13_weight_global_scale",
            "w2_weight_global_scale",
            "w13_input_global_scale",
            "w2_input_global_scale",
        ):
            if hasattr(layer, name):
                delattr(layer, name)
        cfg = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            group_size=self.group_size,
        )
        method = ModelOptNvFp4SM70MoEMethod(quant_config=cfg, moe_config=self.moe)
        layer.base_quant_method = method
        layer._replace_quant_method(method)
        method.process_weights_after_loading(layer)
        logger.info_once(
            "SM70 compressed-tensors NVFP4 MoE bridged to the ModelOpt "
            "TurboMind/QPN route (VLLM_SM70_CT_NVFP4_TM=1); Marlin not used."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
