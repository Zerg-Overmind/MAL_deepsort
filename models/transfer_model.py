import torch
from collections import OrderedDict

model_name = './resnet50/mal_r-50-fpn_cocomodel_0090000.pth'
transferred_model_name = './resnet50/transferred_model_0090000.pth'
#model_name = './resnext101/model_0180000.pth'
#transferred_model_name = './resnext101/transferred_model_0180000.pth'

param_dict_freeanchor = torch.load(model_name)
weights_freeanchor = param_dict_freeanchor['model']

###################### key_map: free_anchor --> nvidia #############################################
original_keys = sorted(param_dict_freeanchor['model'].keys())
layer_keys = sorted(param_dict_freeanchor['model'].keys())



# head
layer_keys = [k.replace("module.rpn.head.cls_tower", "module.cls_head") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.cls_logits", "module.cls_head.8") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.bbox_tower", "module.box_head") for k in layer_keys]
layer_keys = [k.replace("module.rpn.head.bbox_pred", "module.box_head.8") for k in layer_keys]

layer_keys = [k.replace("module.", "") for k in layer_keys]
layer_keys = [k.replace("backbone", "backbones") for k in layer_keys]

key_map = {k: v for k, v in zip(original_keys, layer_keys)}


new_weights = OrderedDict()
count = 0
for k in original_keys:
    if 'anchor_generator' in k:
        continue
    v_freeanchor = weights_freeanchor[k]
    new_weights[key_map[k]] = v_freeanchor

torch.save(new_weights, transferred_model_name)

print('done')
