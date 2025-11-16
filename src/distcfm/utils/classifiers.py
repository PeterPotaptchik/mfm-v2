
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision.transforms import Normalize, Compose, Resize
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # 32x28x28
        x = F.relu(self.conv2(x))       # 64x28x28
        x = self.pool(x)                # 64x14x14
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                 # logits
        return x

def classifier_log_probs(model, x, y):
    """
    Classifier function
    x: Input tensor (images)
    y: Target labels (class idxs)
    """
    logits = model(x) # [B, C]
    log_probs = F.log_softmax(logits, dim=1) # [B, C]
    log_probs = log_probs[range(log_probs.size(0)), y] # [B]
    return log_probs

# https://github.com/YWolfeee/Training-Free-Guidance/blob/main/tasks/networks/huggingface_classifier.py

def load_classifier(dataset, weight_path, device):
    if dataset == "mnist":
        net = MNISTCNN()
        ckpt = torch.load(weight_path,
                          map_location="cpu", 
                          weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        net.to(device)
        transform = lambda x: (x - 0.1307) / 0.3081
        model_fn = lambda x: net(transform(x))
        classifier_fn = lambda x, y: classifier_log_probs(model_fn, x, y)
        return model_fn, classifier_fn
    elif dataset == "mnist_multiclass":
        net = MNISTCNN()
        ckpt = torch.load(weight_path,
                          map_location="cpu", 
                          weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        net.to(device)
        transform = lambda x: (x - 0.1307) / 0.3081
        model_fn = lambda x: net(transform(x))
        def classifier_fn(x, ys):
            logits = model_fn(x)
            log_prob = torch.sum(F.softmax(logits, dim=1) * ys, dim=1).log()
            return log_prob
        return model_fn, classifier_fn
    elif dataset == "cifar10":
        net = AutoModelForImageClassification.from_pretrained(weight_path).to(device)
        processor = AutoImageProcessor.from_pretrained(weight_path)
        try:
            H, W = processor.size['height'], processor.size['width']
        except:
            SIZE = processor.size['shortest_edge']
            H, W = SIZE, SIZE

        MEAN, STD = processor.image_mean, processor.image_std
        transforms = Compose([
            Resize([H, W]),
            Normalize(mean=MEAN, std=STD)
        ])
        model_fn = lambda x: net(transforms(x)).logits
        classifier_fn = lambda x, y: classifier_log_probs(model_fn, x, y)
        return model_fn, classifier_fn
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    