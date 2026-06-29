import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import loralib as lora
from torchvision.models import resnet50, ResNet50_Weights
from torch.distributions import Categorical, Bernoulli


def load_agent(config):
    if config["action_space"] == "standard":
        return REINFORCE(config).to(config["device"])
    elif config["action_space"] == "reg":
        return RL_reg(config).to(config["device"])
    else:
        raise ValueError(f"Invalid action space: {config['action_space']}")

class REINFORCE(nn.Module):
    def __init__(self,config):
        super(REINFORCE, self).__init__()
        self.data = []
        self.r = []
        self.prob = []
        self.model = config["model"]
        # Reward nomalization
        # self.use_norm = config["use_norm"]
        # self.norm_max = 0.0
        # Convolutional layers
        self.conv1 = nn.Conv2d( config["RGB"] , 32, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.relu = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear((config["img_size_x"] // 4) * (config["img_size_y"] // 4) * 64, 512)

        
        # Action mean and log standard deviation layers
        self.action_mean = torch.nn.Linear(512, (config["action_dim"]))
        self.action_logstd = torch.nn.Linear(512, (config["action_dim"]))

        

        self.device = config["device"]

        # Optimizer
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=config["RL_learning_rate"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["RL_learning_rate"])

    def forward(self, state):
        # Forward pass through the network
        image = state.to(self.device)
        image = self.conv1(image)
        image = self.relu(image)
        image = F.max_pool2d(image, 2)
        image = self.conv2(image)
        image = self.relu(image)
        image = F.max_pool2d(image, 2)
        image = torch.flatten(image, 1)
        image = self.fc(image)
        image = self.relu(image)
        action_mean = self.action_mean(image)
        action_logstd = self.action_logstd(image)

        # Compute action standard deviation
        if self.model == "yolo" or self.model == "ddq":
            action_std = 2*torch.sigmoid(action_logstd)
        else:
            action_std = torch.exp(action_logstd)
        return action_mean, action_std


    def train_net(self):
        # Training the network
        self.optimizer.zero_grad()
        loss = -1* self.prob.to(self.device) * self.r/len(self.r)
        # print(loss.sum())
        loss.sum().backward()
        self.optimizer.step()
        self.r,self.prob = [] , []

class RL_reg(nn.Module):
    def __init__(self, config):
        super(RL_reg, self).__init__()
        # 3   4   FCN
        # Channel 0: Coordinate Logits
        # Channel 1, 2, 3: R, G, B Binary Logits
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1) #   4 
        )
        self.sample_ratio = config["attack_pixel"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["RL_learning_rate"])
        self.device = config["device"]

    def forward(self, x, mask, meta_list):
        """
        x: (Total_Objects, 3, 256, 256)
        mask: (Total_Objects, 1, 256, 256)
        meta_list: [[obj1_meta, obj2_meta], [obj3_meta], ...] ( )
        """
        # 1.      1:1 
        # [[obj1, obj2], [obj3]] -> [obj1, obj2, obj3]
        flat_meta = [obj for img_meta in meta_list for obj in img_meta]
        
        total_objects = x.size(0)
        _, _, net_h, net_w = x.size() #   (256, 256)
        
        #       
        assert total_objects == len(flat_meta), "Batch size     !"

        output = self.network(x) # (Total_Objects, 4, 256, 256)
        
        batch_coords = []
        batch_rgb_actions = []
        batch_total_log_probs = []

        # 2.       
        for i in range(total_objects):
            # i   BBox  

            
            # 0.05%     ( 1)
            num_samples = flat_meta[i]['attack_pixels']
            if num_samples == 0:
                batch_total_log_probs.append(torch.tensor(0.0).to(self.device))
                continue
            
            # ---   (Where) ---
            #    
            coord_logits = output[i, 0, :, :].view(1, -1)
            m_flat = mask[i].view(1, -1)
            inf_mask = (1 - m_flat) * -1e10
            probs = torch.softmax(coord_logits + inf_mask, dim=-1)
            
            #  num_samples   ( )
            # action_indices: (1, num_samples)
            action_indices = torch.multinomial(probs, num_samples=num_samples, replacement=False)

            # --- RGB  (What) ---
            rgb_logits_map = output[i, 1:, :, :].view(3, -1) # (3, H*W)
            gather_idx = action_indices.expand(3, -1) # (3, num_samples)
            selected_rgb_logits = rgb_logits_map.gather(1, gather_idx).transpose(0, 1) # (num_samples, 3)
            
            rgb_dist = Bernoulli(logits=selected_rgb_logits)
            rgb_actions = rgb_dist.sample() # (num_samples, 3)

            # ---       ---
            x_c = action_indices % net_w
            y_c = action_indices // net_w
            coords = torch.stack([x_c, y_c], dim=2).squeeze(0) # (num_samples, 2)

            #    (  + RGB )
            coord_log_p = torch.log(probs.gather(1, action_indices) + 1e-10).sum()
            rgb_log_p = rgb_dist.log_prob(rgb_actions).sum()
            
            batch_coords.append(coords) #   ( )
            batch_rgb_actions.append(rgb_actions)
            batch_total_log_probs.append(coord_log_p + rgb_log_p)

        

        #        
        return batch_coords, batch_rgb_actions, torch.stack(batch_total_log_probs)

    def train_net(self):
        # Training the network
        self.optimizer.zero_grad()
        if (
            (not torch.is_tensor(self.prob))
            or self.prob.numel() == 0
            or (not self.prob.requires_grad)
            ):
            self.r, self.prob = [], []
            return
        loss = -1* self.prob.to(self.device) * self.r/len(self.r)
        # print(loss.sum())
        loss.sum().backward()
        self.optimizer.step()
        self.r,self.prob = [] , []


def get_resnet_backbone(out_dim=2048, pretrained=False):
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    modules = list(net.children())[:-1]     # conv1~layer4 + avgpool
    backbone = nn.Sequential(*modules)      #  shape (B,2048,1,1)
    backbone.out_dim = out_dim
    return backbone

def _as_int(x):
    """(k,k) → k   |   k → k"""
    return x[0] if isinstance(x, tuple) else x

def _get_base_conv(lo):
    """
    lora.Conv2d    Conv     
    -  loralib: lo.conv
    -     : lo.base_layer
    - Fallback    : lo  
    """
    if hasattr(lo, "conv"):          # loralib ≥ 0.2
        return lo.conv
    if hasattr(lo, "base_layer"):    # loralib ≤ 0.1
        return lo.base_layer
    return lo                        #  

def loraize_cnn(module, r=4, alpha=16, dropout=0, freeze_base=True):
    """
     nn.Conv2d lora.Conv2d .
    freeze_base=True ⇒   W , LoRA ΔW 
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            lo = lora.Conv2d(
                in_channels   = child.in_channels,
                out_channels  = child.out_channels,
                kernel_size   = _as_int(child.kernel_size),
                stride        = _as_int(child.stride),
                padding       = _as_int(child.padding),
                dilation      = _as_int(child.dilation),
                groups        = child.groups,
                bias          = (child.bias is not None),
                r             = r,
                lora_alpha    = alpha,
                lora_dropout  = dropout
            )
            # 2)  · 
            base = _get_base_conv(lo)
            base.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                base.bias.data.copy_(child.bias.data)

            # 3) freeze  ( )
            base.weight.requires_grad_(not freeze_base)
            if child.bias is not None:
                base.bias.requires_grad_(not freeze_base)

            setattr(module, name, lo)
        else:
            loraize_cnn(child, r, alpha, dropout, freeze_base)



# class ResNetLoRA_RL(nn.Module):
#     def __init__(self, cfg, backbone):
#         super().__init__()
#         self.backbone = backbone
#         self.device   = cfg["device"]
#         self.model_name = cfg["model"]

#         fc_in = backbone.out_dim       #  2048
#         self.fc      = nn.Linear(fc_in, 512)
#         self.policy  = nn.Linear(512, cfg["action_dim"])
#         self.log_std = nn.Linear(512, cfg["action_dim"])
#         # b
#         # self.value = nn.Linear(512, 1)  # b(s) ≈ V(s)
#         #
#         if cfg["use_lora"] == True:
#             self.freeze_except_trainables()
#         else:
#             self.unfreeze_all()
#         self.optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, self.parameters()),
#             lr=cfg["RL_learning_rate"]
#         )
#         #
#         self.r, self.prob, self.norm_max = [], [], 0.0
#         #
#     # ─────────────   ─────────────
#     def unfreeze_all(self):
#         for n, p in self.named_parameters():
#             p.requires_grad_(True)
#     def freeze_except_trainables(self):
#         for n, p in self.named_parameters():
#             if (
#                 "lora_" in n
#                 or n.startswith("fc")
#                 or n.startswith(("policy", "log_std"))
#                 # or n.startswith(("value", "policy", "log_std"))
#             ):
#                 p.requires_grad_(True)
#             else:
#                 p.requires_grad_(False)

#     # ───────────── forward ─────────────
#     def forward(self, x):
#         x = self.backbone(x.to(self.device))          # (B,2048,1,1)
#         x = x.flatten(1)
#         x = torch.relu(self.fc(x))
#         mean, log_s = self.policy(x), self.log_std(x)
#         # mean, log_s, self.v = self.policy(x), self.log_std(x), self.value(x).squeeze()
#         std = 2*torch.sigmoid(log_s) if self.model_name in {"yolo","ddq"} \
#               else torch.exp(log_s)
#         return mean, std

#     # ───────────── REINFORCE  ─────────────
#     def train_net(self):
#         self.optimizer.zero_grad()

#         if (
#             (not torch.is_tensor(self.prob))
#             or self.prob.numel() == 0
#             or (not self.prob.requires_grad)
#         ):
#             self.r, self.prob = [], []
#             return

#         loss = -(self.prob.to(self.device) * (self.r) / len(self.r)).sum()

#         # loss = -(self.prob.to(self.device) * (self.r-self.v.detach()) / len(self.r)).sum()
#         # value_loss = nn.functional.mse_loss(self.v, self.r)
#         # loss = loss + value_loss
#         loss.backward()
#         self.optimizer.step()
#         self.r, self.prob = [], []

#     # ───────────── LoRA·  ─────────────
#     def init_lora_and_head(self, lora_init="kaiming", logstd_bias=0):
#         for m in self.modules():
#             if hasattr(m, "lora_A") and m.lora_A is not None:
#                 if lora_init == "kaiming":
#                     nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
#                 else:
#                     nn.init.xavier_uniform_(m.lora_A)
#                 nn.init.zeros_(m.lora_B)
#         nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.fc.bias)
#         nn.init.xavier_uniform_(self.policy.weight)
#         nn.init.zeros_(self.policy.bias)
#         nn.init.xavier_uniform_(self.log_std.weight)
#         nn.init.constant_(self.log_std.bias, logstd_bias)

