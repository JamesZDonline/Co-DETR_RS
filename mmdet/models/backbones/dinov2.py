
import torch
import torch.nn as nn
from ..builder import BACKBONES
from dinov2.models.vision_transformer import vit_large


@BACKBONES.register_module()
class CustomDinoViTV2(nn.Module):
    def __init__(self,pretrained_weight_path):
        super(CustomDinoViTV2,self).__init__()
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        teacher_checkpoint = pretrained_weight_path # path to .pth
        pretrained_dict = torch.load(teacher_checkpoint)
        checkpoint_key = 'teacher'
        new_state_dict = {}
        for k, v in pretrained_dict[checkpoint_key].items():
            if 'dino_head' in k:
                print(f'{k} not used')    
            elif 'ibot_head' in k:
                print(f'{k} not used')
            else:
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
        #change shape of pos_embed, shape depending on vits or vitg, or  vitl
        pos_embed = nn.Parameter(torch.zeros(1, 257, 1024))
        self.dino_model.pos_embed = pos_embed# change shape of patch embed (it was not all correct)
        # patch_embed_weight = nn.Parameter(torch.zeros(1024, 8, 14, 14))
        # model.patch_embed.proj.weight = patch_embed_weight########################################################################
        new_patch_embed = self.dino_model.patch_embed
        new_patch_embed.proj = nn.Conv2d(
            in_channels=8,  # Updated for 8 input bands
            out_channels=new_patch_embed.proj.out_channels,
            kernel_size=new_patch_embed.proj.kernel_size,
            stride=new_patch_embed.proj.stride,
            padding=new_patch_embed.proj.padding,
            # bias=new_patch_embed.proj.bias,
        )
        # Replace the old PatchEmbed with the updated one
        self.dino_model.patch_embed = new_patch_embed# load state dict
        self.dino_model.load_state_dict(new_state_dict, strict=True)


    def forward(self, x):
        output = self.dino_model.forward_features(x)
        print(output['x_norm_patchtokens'].shape)
        output = output['x_norm_patchtokens'].reshape(1, 16, 16, 1024)
        outputs = [output.permute(0, 3, 1, 2).contiguous()]
        return outputs



