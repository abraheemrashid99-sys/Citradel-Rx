"""NR channel coding of the data part (reviewer audit D2).

Every code block carries `k - 24` payload bits and a CRC-24A (3GPP TS 38.212 Sec. 5.1) and is
encoded with the 5G LDPC code (TS 38.212 Sec. 5.3.2, base graph chosen by the standard's rule, rate
matching to `n` bits) by Sionna's `LDPC5GEncoder` (Sionna PHY 2.2, PyTorch). check() decodes
hard symbol decisions (belief propagation, 20 iterations, channel LLR magnitude 3 for every bit)
and returns the CRC verdict per code block and user, in the place of the CRC-8 sub-blocks: the
receivers' gates, the CRC selection and the BLER then work on decoded code blocks.
"""
from __future__ import annotations

import numpy as np
import torch


class LDPCFramer:
    def __init__(self, k: int, n: int, bits_per_symbol: int, num_iter: int = 20, llr_mag: float = 3.0):
        from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
        from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
        if n % bits_per_symbol:
            raise ValueError("coded bits per code block must fill whole symbols")
        self.enabled = True
        self.r, self.k, self.n, self.bps = 24, k, n, bits_per_symbol
        self.L = n // bits_per_symbol           # symbols per code block
        self.sub_bits = n
        self.payload_bits = k - 24
        self.llr_mag = llr_mag
        self.crc = CRCEncoder("CRC24A")
        self.crc_dec = CRCDecoder(self.crc)
        self.enc = LDPC5GEncoder(k, n)
        self.dec = LDPC5GDecoder(self.enc, num_iter=num_iter, hard_out=True, return_infobits=True)

    def frame(self, payload: np.ndarray) -> np.ndarray:
        """payload (n_cb, n_user, k - 24) -> coded bits (n_cb, n_user, n)."""
        with torch.no_grad():
            c = self.enc(self.crc(torch.as_tensor(payload, dtype=torch.float32)))
        return c.numpy().astype(np.int64)

    def n_subblocks(self, data_bits: int) -> int:
        return data_bits // self.n

    def check(self, decided_bits: np.ndarray) -> np.ndarray:
        """decided bits (n_cb, n_user, n) -> CRC pass after LDPC decoding (n_cb, n_user)."""
        b = torch.as_tensor(decided_bits, dtype=torch.float32)
        with torch.no_grad():
            u = self.dec((2.0 * b - 1.0) * self.llr_mag)      # logit convention: positive = bit 1
            _, ok = self.crc_dec(u)
        return ok.squeeze(-1).numpy().astype(bool)
