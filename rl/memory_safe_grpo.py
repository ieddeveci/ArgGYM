from __future__ import annotations

import torch

from trl import GRPOTrainer
from trl.trainer.utils import selective_log_softmax


class MemorySafeGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer variant that avoids materializing the full
    [completion_length, vocab_size] logits tensor when computing
    old-policy/reference token log-probabilities.

    The differentiable policy loss is expected to use TRL's Liger
    FusedLinearGRPOLoss via use_liger_kernel=True.

    This override only takes the memory-safe path when Liger is enabled
    and entropy/auxiliary-loss computation is not requested. Other calls
    fall back to upstream TRL behavior.
    """

    def __init__(
        self,
        *args,
        logprob_token_chunk_size: int = 256,
        liger_loss_compiled: bool = False,
        **kwargs,
    ):
        self.logprob_token_chunk_size = int(logprob_token_chunk_size)
        if self.logprob_token_chunk_size <= 0:
            raise ValueError("logprob_token_chunk_size must be > 0")

        self.liger_loss_compiled = bool(liger_loss_compiled)

        super().__init__(*args, **kwargs)

        if self.use_liger_kernel:
            liger_loss = getattr(self, "liger_loss", None)
            if liger_loss is None:
                raise RuntimeError(
                    "use_liger_kernel=True but trainer has no liger_loss"
                )
            if not hasattr(liger_loss, "compiled"):
                raise RuntimeError(
                    "TRL liger_loss has no 'compiled' attribute; "
                    "refusing to silently continue"
                )

            liger_loss.compiled = self.liger_loss_compiled

            print(
                "ArgGYM Liger GRPO loss compile resolved: "
                f"compiled={liger_loss.compiled}",
                flush=True,
            )

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        # Keep upstream behavior for any path for which the fused/chunked
        # implementation has not explicitly been validated.
        if (
            not self.use_liger_kernel
            or compute_entropy
            or compute_aux_loss
            or pixel_values is not None
            or image_grid_thw is not None
            or num_images is not None
            or pixel_attention_mask is not None
            or spatial_shapes is not None
            or num_tiles is not None
            or image_sizes is not None
            or token_type_ids is not None
            or mm_token_type_ids is not None
            or image_position_ids is not None
        ):
            return super()._get_per_token_logps_and_entropies(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                num_tiles=num_tiles,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_position_ids=image_position_ids,
            )

        if not getattr(self, "_memory_safe_logprob_announced", False):
            print(
                "ArgGYM memory-safe old-logprob path active: "
                f"token_chunk_size={self.logprob_token_chunk_size}",
                flush=True,
            )
            self._memory_safe_logprob_announced = True

        batch_size = batch_size or input_ids.size(0)
        all_logps = []

        # We deliberately bypass the Accelerate model.forward FP32-output
        # wrapper, since that wrapper is what causes the complete dense
        # logits tensor to be promoted to FP32 at once.
        unwrapped_model = self.accelerator.unwrap_model(model)
        lm_head = unwrapped_model.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("Model has no output embedding / lm_head")

        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))

            ids_batch = input_ids[start:end]
            mask_batch = attention_mask[start:end]

            # Preserve the same autocast regime as normal trainer forwards.
            with self.accelerator.autocast():
                hidden = self._get_last_hidden_state(
                    unwrapped_model,
                    ids_batch,
                    mask_batch,
                    logits_to_keep,
                )

            completion_ids = ids_batch[:, -logits_to_keep:]
            chunk_logps = []

            for tok_start in range(
                0,
                logits_to_keep,
                self.logprob_token_chunk_size,
            ):
                tok_end = min(
                    tok_start + self.logprob_token_chunk_size,
                    logits_to_keep,
                )

                hidden_chunk = hidden[:, tok_start:tok_end, :]
                token_ids_chunk = completion_ids[:, tok_start:tok_end]

                # Match the original path:
                #   BF16/autocast LM-head GEMM
                #   -> FP32 output
                #   -> temperature
                #   -> selective log-softmax
                with self.accelerator.autocast():
                    logits_chunk = lm_head(hidden_chunk)

                logits_chunk = logits_chunk.float()
                logits_chunk = logits_chunk / self.temperature

                selected_logps = selective_log_softmax(
                    logits_chunk,
                    token_ids_chunk,
                )
                chunk_logps.append(selected_logps)

                # Make lifetime explicit; important for very long traces.
                del logits_chunk

            all_logps.append(torch.cat(chunk_logps, dim=1))

        return torch.cat(all_logps, dim=0), None, None
