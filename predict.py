import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model file')
    parser.add_argument('--image', type=str, help='image to test')
    opt = parser.parse_args()
    modelPath = opt.model
    image_path = opt.image
    transform_normalize_mean = (0.5, 0.5, 0.5)
    transform_normalize_var = (0.5, 0.5, 0.5)

    transform = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize(transform_normalize_mean, transform_normalize_var),
        ]
    )

    device = 'cpu'
    model = torch.jit.load(modelPath)
    model.eval()
    model.to(device)
    img = Image.open(image_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img.to(device)

    model.eval()
    prediction = model(img)
    prediction = prediction.argmax()
    print(f"Image belongs to class {prediction}")

