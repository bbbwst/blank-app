import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# Define UNet components
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x, return_attention=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if return_attention:
            return logits, x5
        return logits

# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=1, n_classes=2, bilinear=True)
model.load_state_dict(torch.load("my_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🫁 Lung Nodule Segmentation with UNet")

# Sample slider for images in Sample/
import os
sample_dir = "Sample"
sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if sample_files:
    selected_file = st.selectbox("Select a sample image to preview", sample_files)
    sample_path = os.path.join(sample_dir, selected_file)
    st.image(sample_path, caption=f"Sample: {selected_file}", use_container_width=True)
    use_sample = st.checkbox("Use this sample for prediction", value=False)
else:
    use_sample = False

uploaded_file = st.file_uploader("Upload a CT scan slice (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None or (use_sample and sample_files):
    if use_sample and sample_files:
        image = Image.open(sample_path).convert("L")
        st.image(image, caption=f"Sample: {os.path.basename(sample_path)}", use_container_width=True)
    else:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)
        pred_mask = (pred > 0.5).float().cpu().numpy()
        # If pred_mask has shape (batch, channels, H, W), select the first batch and class/channel
        if pred_mask.ndim == 4:
            pred_mask = pred_mask[0]
        if pred_mask.shape[0] > 1:
            # For multi-class, select the foreground class (e.g., class 1)
            pred_mask = pred_mask[1]
        else:
            pred_mask = pred_mask[0]

    # Overlay mask in red on the original image
    orig_img = image.resize((256, 256)).convert("RGB")
    mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    mask_rgba[..., 0] = (pred_mask * 255).astype(np.uint8)  # Red channel
    mask_rgba[..., 3] = (pred_mask * 220).astype(np.uint8)  # Alpha channel for stronger red
    mask_overlay = Image.fromarray(mask_rgba, mode="RGBA")
    overlayed = orig_img.copy()
    overlayed.paste(mask_overlay, (0, 0), mask_overlay)
    st.image(overlayed, caption="Predicted Mask (Red Overlay)", use_container_width=True)
