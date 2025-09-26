import argparse
import torch
import torchvision as tv
from PIL import Image
from model import make_model

IMG_TF = tv.transforms.Compose([
    tv.transforms.Resize((256,256)),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    return IMG_TF(img).unsqueeze(0)


def predict_image(model, path, device):
    x = load_image(path).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
    return pred


def main():
    p = argparse.ArgumentParser(description="Predict chirality (left/right) for one or more images")
    p.add_argument("image", nargs="+", help="Path(s) to image file(s)")
    p.add_argument("--weights", default="best_chirality.pt", help="Path to model weights")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model()
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    labels = ["left", "right", "nonchiral"]
    for ip in args.image:
        if not ip or not ip.strip():
            continue
        if not os.path.exists(ip):
            print(f"[WARN] file not found: {ip}")
            continue
        pred = predict_image(model, ip, device)
        label = labels[pred] if pred < len(labels) else str(pred)
        print(f"{ip}\t->\t{label}")


if __name__ == "__main__":
    import os
    main()
