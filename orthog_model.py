from timm import create_model
from torch import nn
from torch import transpose, bmm
from torch.nn import AvgPool2d, LayerNorm, Dropout


class orthog_model(nn.Module):
    def __init__(self, orthog=True, model_name: str = None, model=None, dropout_rate=None, num_labels=1000,
                 feature_reduction=16):
        super().__init__()
        self.orthog = orthog
        self.num_labels = num_labels
        self.model_name = model_name
        if model_name is not None and model is None:
            if "vitb16" in model_name:
                fe1 = create_model("vit_base_patch16_224")
                if orthog:
                    fe2 = create_model("vit_base_patch16_224")
            if "vitb32" in model_name:
                fe1 = create_model("vit_base_patch32_224")
                if orthog:
                    fe2 = create_model("vit_base_patch32_224")
            if "vitl16" in model_name:
                fe1 = create_model("vit_large_patch16_224")
                if orthog:
                    fe2 = create_model("vit_large_patch16_224")
            if "vitl32" in model_name:
                fe1 = create_model("vit_large_patch32_224")
                if orthog:
                    fe2 = create_model("vit_large_patch32_224")
            if "vits16" in model_name:
                fe1 = create_model("vit_small_patch16_224")
                if orthog:
                    fe2 = create_model("vit_small_patch16_224")
            if "vits32" in model_name:
                fe1 = create_model("vit_small_patch32_224")
                if orthog:
                    fe2 = create_model("vit_small_patch32_224")

            self.average_layer = AvgPool2d((feature_reduction, 1), stride=(feature_reduction, 1), padding=0)
            self.layernorm = LayerNorm(fe1.embed_dim)
            self.dropout_rate = dropout_rate
            if self.dropout_rate is not None:
                self.dp = Dropout(self.dropout_rate)
            self.classifier = nn.Linear(fe1.embed_dim, self.num_labels)
            self.fe1 = fe1
            if orthog:
                self.fe2 = fe2

    def forward_features(self, x):
        if "vit" in self.model_name:
            encoder_outputs = self.fe1.forward_features(x)[:, 0][:, None]
            struct = self.fe2.forward_features(x)
            struct = self.average_layer(struct[:, 1:])
            proj = bmm(transpose(struct, 1, 2), struct)
            sequence_output = encoder_outputs - bmm(encoder_outputs, proj)
            return sequence_output

    def forward(self, x):
        if self.orthog:
            sequence_output = self.forward_features(x)
            out = self.layernorm(sequence_output)
            if self.dropout_rate is not None:
                out = self.dp(out)
            logit = self.classifier(out)
        else:
            logit = self.fe1(x)
        return logit
