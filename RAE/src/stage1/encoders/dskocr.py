from torch import nn
import torch
from math import *
from . import register_encoder
import ipdb
st = ipdb.set_trace


@register_encoder()
class DeepSeekOCR(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # from transformers import Dinov2WithRegistersModel
        # Support both local paths and HuggingFace model IDs
        # try:
        #     self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        # except (OSError, ValueError, AttributeError):
        #     self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.patch_size = 64
        self.hidden_size = 1280
        # st()
        model_name = "/home/mprabhud/phd_projects/continuous_diffusion/DeepSeek-OCR-code"  # Commented out to use HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
        # model = model.eval().cuda().to(torch.bfloat16)           
        self.encoder.requires_grad_(False)
        # st()
        # if normalize:
        #     self.encoder.layernorm.elementwise_affine = False
        #     self.encoder.layernorm.weight = None
        #     self.encoder.layernorm.bias = None
        # self.patch_size = self.encoder.config.patch_size
        # self.hidden_size = self.encoder.config.hidden_size
        
    # def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.encoder(x, output_hidden_states=True)
    #     unused_token_num = 5  # 1 CLS + 4 register tokens
    #     image_features = x.last_hidden_state[:, unused_token_num:]
    #     return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        image_features = self.encoder.get_image_features(x)
        return image_features
