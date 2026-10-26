"""MINT-GR data-shape contract shared by formal and smoke entrypoints."""

import torch
from torch.nn.utils.rnn import pad_sequence


def atomic_history_collator_length(config) -> int:
    """Translate a max atomic history into the legacy Collator convention.

    ``Collator.__pad_seq__`` retains ``max_length - 1`` items when truncating,
    because the original pipeline reserved one slot for a target/eos token.
    MINT-GR passes only histories, so a configured history limit H requires
    ``H + 1`` here.
    """
    maximum = int(config.get("max_his_len", 50))
    if maximum < 1:
        raise ValueError("max_his_len must be positive")
    return maximum + 1


class MintAtomicHistoryCollator:
    """Pad atomic item histories with an exact, suffix-preserving limit."""

    def __init__(self, max_history: int, padding_id: int = 0):
        self.max_history = int(max_history)
        self.padding_id = int(padding_id)
        if self.max_history < 1:
            raise ValueError("max_history must be positive")

    def __call__(self, batch):
        histories, targets = zip(*batch)
        tensors = [
            torch.tensor(history[-self.max_history :], dtype=torch.long)
            for history in histories
        ]
        input_ids = pad_sequence(
            tensors, batch_first=True, padding_value=self.padding_id
        )
        attention_mask = input_ids.ne(self.padding_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": torch.tensor(targets, dtype=torch.long).contiguous(),
        }
