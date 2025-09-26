import torch, torchvision as tv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random, os

# add this top-level function so it's picklable on Windows
def _identity(x):
    return x

class ChiralityDataset(Dataset):
    def __init__(self, root, split="train"):
        # Expect folder structure: root/left/*.jpg, root/right/*.jpg, root/nonchiral/*.jpg
        # third class (index 2) is nonchiral examples
        self.items = []
        for label, cls in enumerate(["left", "right", "nonchiral"]):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                # skip missing class directories (user may not have nonchiral examples yet)
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg",".png",".jpeg")):
                    self.items.append((os.path.join(cls_dir, f), label))
        self.train = (split == "train")
        self.base_tf = tv.transforms.Compose([
            tv.transforms.Resize((256,256)),
            tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if self.train else tv.transforms.CenterCrop(224),
            tv.transforms.ColorJitter(0.1,0.1,0.1,0.05) if self.train else tv.transforms.Lambda(_identity),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        # Chirality-aware augmentation: randomly mirror *and* flip the label for left/right only
        if self.train and random.random() < 0.5:
            img = tv.transforms.functional.hflip(img)
            # only swap label for left(0)<->right(1). nonchiral (2) stays the same
            if y in (0, 1):
                y = 1 - y
        img = self.base_tf(img)
        return img, y

def get_loader(root, split, bs):
    ds = ChiralityDataset(root, split)
    return DataLoader(ds, batch_size=bs, shuffle=(split=="train"), num_workers=4, pin_memory=True)

def make_model():
    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    # now supporting three classes: left, right, nonchiral
    m.fc = nn.Linear(m.fc.in_features, 3)
    return m

def train_one_epoch(model, opt, loader, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward(); opt.step()
        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.numel()
    return correct/total

@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    from sklearn.metrics import balanced_accuracy_score
    ys, ps = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        ps += logits.argmax(1).cpu().tolist()
        ys += y.tolist()
    return balanced_accuracy_score(ys, ps)

# add this new helper to compute and print a confusion matrix + per-class report
@torch.no_grad()
def eval_confusion(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        ps += logits.argmax(1).cpu().tolist()
        ys += y.tolist()

    from sklearn.metrics import confusion_matrix, classification_report
    unique_labels = sorted(set(ys))
    cm = confusion_matrix(ys, ps, labels=unique_labels)
    label_names = ["left", "right", "nonchiral"]
    names = [label_names[i] if i < len(label_names) else str(i) for i in unique_labels]

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class metrics:")
    print(classification_report(ys, ps, labels=unique_labels, target_names=names, zero_division=0))
    return cm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = get_loader("data/train", "train", 64)
    val_loader   = get_loader("data/val", "val", 64)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best, patience, wait = 0.0, 10, 0
    for epoch in range(100):
        tr_acc = train_one_epoch(model, opt, train_loader, device)
        val_bal_acc = eval_metrics(model, val_loader, device)
        print(f"epoch {epoch} train_acc={tr_acc:.3f} val_bal_acc={val_bal_acc:.3f}")
        # print confusion matrix / report for the validation set
        eval_confusion(model, val_loader, device)
        if val_bal_acc > best:
            best, wait = val_bal_acc, 0
            torch.save(model.state_dict(), "best_chirality.pt")
        else:
            wait += 1
            if wait >= patience: break
