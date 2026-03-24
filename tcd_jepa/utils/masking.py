"""Multi-block masking for JEPA training. Adapted from I-JEPA (Meta Platforms, Inc.)."""

import math
from logging import getLogger
from multiprocessing import Value
from typing import Optional

import torch

logger = getLogger(__name__)


class MaskCollator:
    """Generates context and predictor masks for JEPA training.

    Produces block-shaped masks for both the encoder (context) and predictor (target).
    Context masks are typically smaller blocks; target masks are larger blocks.
    Adapted from I-JEPA's MaskCollator.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        enc_mask_scale: tuple[float, float] = (0.15, 0.2),
        pred_mask_scale: tuple[float, float] = (0.2, 0.4),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        nenc: int = 4,
        npred: int = 1,
        min_keep: int = 10,
        allow_overlap: bool = False,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        """Thread-safe iteration counter."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect_ratio_scale: tuple[float, float],
    ) -> tuple[int, int]:
        """Sample a block size given scale and aspect ratio ranges."""
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        h = max(h, 1)
        w = max(w, 1)
        return (h, w)

    def _sample_block_mask(
        self,
        b_size: tuple[int, int],
        acceptable_regions: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a block mask and its complement."""
        h, w = b_size

        def constrain_mask(mask: torch.Tensor, tries: int = 0) -> None:
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, max(self.height - h, 1), (1,))
            left = torch.randint(0, max(self.width - w, 1), (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) >= self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f"Valid mask not found, decreasing acceptable-regions [{tries}]"
                    )
                    max_tries = len(acceptable_regions) + 1 if acceptable_regions is not None else 5
                    if tries > max_tries:
                        # All acceptable regions exhausted or too many retries;
                        # fall back to unconstrained mask
                        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
                        mask[top : top + h, left : left + w] = 1
                        mask = torch.nonzero(mask.flatten())
                        if len(mask) == 0:
                            # Absolute fallback: use all patches
                            mask = torch.arange(self.height * self.width).unsqueeze(-1)
                        break
        mask = mask.squeeze(-1)

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        return mask, mask_complement

    def __call__(
        self, batch: list
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Create encoder and predictor masks when collating images into a batch."""
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        p_size = self._sample_block_size(
            generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio
        )
        e_size = self._sample_block_size(
            generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0)
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = None if self.allow_overlap else masks_C

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Trim masks to minimum kept size and collate
        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
