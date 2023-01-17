from math import ceil
from dataclasses import dataclass, field
from typing import Union, Optional


Sameseq = Union[int, tuple[int], list[int]]

@dataclass
class ConvArgs:
    in_channels: int = field()
    out_channels: int = field()
    kernel_size: Sameseq = field(default=1)
    stride: Sameseq = field(default=1)
    padding: Sameseq = field(default=None)
    dilation: Sameseq = field(default=1)
    bias: bool = field(default=False)
    groups: bool = field(default=1)

    def __post_init__(self):
        if self.padding is None:
            self.padding = (self.kernel_size - 1) // 2 * self.dilation


@dataclass
class MBconfig:
    expand_ratio: int = field()
    kernel: int = field()
    stride: int = field()
    input_channels: int = field()
    out_channels: int = field()
    num_layers: int = field()
    width_mult: float = field(default=1.0)
    depth_mult: float = field(default=1.0)
    sd_prob: float = field(default=0.2)

    @staticmethod
    def _make_divisible(v: float,
                        divisor: int,
                        min_value: Optional[int] = None
                        ) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @staticmethod
    def adjust_channels(channels: int, width_mult: float,
                        min_value: Optional[int] = None) -> int:
        return MBconfig._make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(ceil(num_layers * depth_mult))

    def __post_init__(self):
        self.input_channels = self.adjust_channels(
            self.input_channels, self.width_mult
        )
        self.out_channels = self.adjust_channels(
            self.out_channels, self.width_mult
        )
        self.num_layers = self.adjust_depth(
            self.num_layers, self.depth_mult
        )