import copy
from fastai.text import *

def seed_all(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def delete_encoding_layers(model, num_layers_to_keep):
    old_module_list = model.transformer.bert.encoder.layer
    new_module_list = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        new_module_list.append(old_module_list[i])

    # create a copy of the model, modify it with the new list, and return
    copy_of_model = copy.deepcopy(model)
    copy_of_model.transformer.bert.encoder.layer = new_module_list

    return copy_of_model


def unpack_tensor(tensor_tuple, classes):
  result_keys = ["category", "tensor", "classes"]
  unpack_dict = {}
  for i in range(len(tensor_tuple)):
    unpack_dict[result_keys[i]] = tensor_tuple[i]
  for i in range(len(classes)):
    unpack_dict[classes[i]] = unpack_dict["classes"][i]
  unpack_dict.pop("classes")
  return unpack_dict
