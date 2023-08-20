import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchLinPositionEmbedding(nn.Module):
    def __init__(self, in_channels: int = 4, emb_size: int = 768, gene_size: int = 14123):
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            # Rearrange('b (p g) c -> b p (g c)'),
            nn.Linear(in_channels, emb_size),
        )
        self.positions = nn.Parameter(torch.randn(gene_size, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x)
        # add position embedding
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.attn = None

    def forward(self, x: Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b g (h d qkv) -> (qkv) b h g d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(torch.div(energy, scaling), dim=-1)
        att = self.att_drop(att)
        self.attn = att
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h g d -> b g (h d)")
        out = self.projection(out)
        return out

# patches_embedded = PatchLinPositionEmbedding()(x)
# MultiHeadAttention()(patches_embedded).shape


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.1,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

# patches_embedded = PatchLinCLSPositionEmbedding()(x)
# TransformerEncoderBlock()(patches_embedded).shape


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super().__init__(
            Reduce('b g e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class MultiomicsT(nn.Sequential):
    def __init__(self,
                in_channels: int = 4,
                emb_size: int = 256,
                gene_size: int = 14123,
                depth: int = 2,
                n_classes: int = 14123,
                drop_p: float = 0.1,
                **kwargs):
        super().__init__(
            PatchLinPositionEmbedding(in_channels, emb_size, gene_size),
            TransformerEncoder(depth, emb_size=emb_size, drop_p=drop_p, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

# summary(MultiomicsT(), (14123, 4), batch_size = 1, device='cpu')