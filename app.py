# -*- coding: utf-8 -*-
"""
Application PyQt6 Unifiée : cGAN pour la Colorisation de Lineart

Ce script Python est une application de bureau complète et autonome pour entraîner
et utiliser un réseau GAN de colorisation, désormais dotée d'un entraînement
progressif pour atteindre de hautes résolutions (jusqu'à 1024px).

AMÉLIORATIONS DE ROBUSTESSE :
- CORRECTION : Correction d'une erreur d'indexation dans la création des indices
  de couleur (Color Hints).
- Inférence adaptative, noms de checkpoints intelligents, et architecture HD.

Dépendances (à installer via pip) :
# pip install torch torchvision torchaudio
# pip install opencv-python-headless
# pip install numpy
# pip install Pillow
# pip install PyQt6
# pip install matplotlib
# pip install tqdm
# pip install scikit-image
# pip install qdarkstyle
"""
import os
import sys
import threading
import time
import traceback
import random
import re

import cv2
import numpy as np
import qdarkstyle
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.ImageQt import ImageQt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGridLayout,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QSlider, QTabWidget,
                             QTextEdit, QVBoxLayout, QWidget, QCheckBox, QSpinBox)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ==============================================================================
# --- SECTION 1: CONFIGURATION
# ==============================================================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count() // 2
    LEARNING_RATE_G = 2e-4
    LEARNING_RATE_D = 1e-4
    BETA1 = 0.5
    BETA2 = 0.999
    BATCH_SIZES = {128: 16, 256: 8, 512: 4, 1024: 1}
    NUM_EPOCHS_PER_STAGE = 50
    L1_LAMBDA = 200.0
    VGG_LAMBDA = 20.0
    FEATURE_MATCHING_LAMBDA = 10.0
    G_TRAIN_STEPS = 2
    NUM_RESIDUALS = 9
    HINT_PHASE_START_RATIO = 0.9
    HINT_DENSITY = 0.005
    ROOT_DIR = "GAN_Colorization_Project"
    RAW_DATA_DIR = os.path.join(ROOT_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "processed_data")
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    SAVE_MODEL_FREQ = 10


for path in [Config.ROOT_DIR, Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(path, exist_ok=True)


# ==============================================================================
# --- SECTION 2: MODELES & LOGIQUE IA
# ==============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, use_activation=True, use_spectral_norm=False,
                 **kwargs):
        super().__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if use_spectral_norm: self.conv = nn.utils.spectral_norm(self.conv)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else None
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x);
        if self.norm: x = self.norm(x)
        if self.use_activation: x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            ConvBlock(channels, channels, use_activation=False, kernel_size=3, padding=1, padding_mode="reflect")
        )

    def forward(self, x): return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=64, num_residuals=9):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, features, 7, 1, 3, padding_mode="reflect"),
                                          nn.InstanceNorm2d(features), nn.ReLU(True))
        self.down1 = ConvBlock(features, features * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = ConvBlock(features * 2, features * 4, kernel_size=3, stride=2, padding=1)
        self.down3 = ConvBlock(features * 4, features * 8, kernel_size=3, stride=2, padding=1)
        self.residuals = nn.Sequential(*[ResidualBlock(features * 8) for _ in range(num_residuals)])
        self.up1 = nn.Sequential(nn.ConvTranspose2d(features * 8, features * 4, 3, 2, 1, 1),
                                 nn.InstanceNorm2d(features * 4), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(features * 4 * 2, features * 2, 3, 2, 1, 1),
                                 nn.InstanceNorm2d(features * 2), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(features * 2 * 2, features, 3, 2, 1, 1),
                                 nn.InstanceNorm2d(features), nn.ReLU(True))
        self.final_up = nn.Sequential(nn.Conv2d(features * 2, out_channels, 7, 1, 3, padding_mode="reflect"), nn.Tanh())

    def forward(self, x):
        d1 = self.initial_down(x);
        d2 = self.down1(d1);
        d3 = self.down2(d2);
        d4 = self.down3(d3)
        b = self.residuals(d4)
        u1 = self.up1(b);
        u2 = self.up2(torch.cat([u1, d3], 1));
        u3 = self.up3(torch.cat([u2, d2], 1))
        return self.final_up(torch.cat([u3, d1], 1))


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult;
            nf_mult = min(2 ** n, 8)
            layers += [ConvBlock(features * nf_mult_prev, features * nf_mult, kernel_size=4, stride=2, padding=1,
                                 use_spectral_norm=True)]
        nf_mult_prev = nf_mult;
        nf_mult = min(2 ** n_layers, 8)
        layers += [ConvBlock(features * nf_mult_prev, features * nf_mult, kernel_size=4, stride=1, padding=1,
                             use_spectral_norm=True)]
        layers += [nn.Conv2d(features * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x, y): return self.model(torch.cat([x, y], dim=1))


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64, n_layers=3, num_d=2):
        super().__init__()
        self.num_d = num_d
        for i in range(num_d):
            setattr(self, 'discriminator_' + str(i), NLayerDiscriminator(in_channels, features, n_layers))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x, y):
        results = []
        for i in range(self.num_d):
            output = getattr(self, 'discriminator_' + str(i))(x, y)
            results.append(output)
            if i != (self.num_d - 1): x = self.downsample(x); y = self.downsample(y)
        return results

    def forward_features(self, x, y):
        results = []
        for i in range(self.num_d):
            netD = getattr(self, 'discriminator_' + str(i))
            model = netD.model;
            intermediate_features = []
            out = torch.cat([x, y], dim=1)
            for layer in model:
                out = layer(out);
                intermediate_features.append(out)
            results.append(intermediate_features)
            if i != (self.num_d - 1): x = self.downsample(x); y = self.downsample(y)
        return results


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:21].eval().to(device)
        for param in vgg19.parameters(): param.requires_grad = False
        self.vgg = vgg19;
        self.loss_fn = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, gen, real):
        gen_norm = (gen - self.mean) / self.std;
        real_norm = (real - self.mean) / self.std
        return self.loss_fn(self.vgg(gen_norm), self.vgg(real_norm))


# CORRECTION : Utilisation des bonnes dimensions pour le tenseur 3D
def create_color_hints(color_image_tensor, density=0.005):
    """Crée une image d'indices de couleur à partir d'une image couleur."""
    hints = torch.zeros_like(color_image_tensor)
    # Un tenseur d'image unique a la forme (C, H, W)
    # On utilise donc les indices 1 (Hauteur) et 2 (Largeur)
    mask = torch.rand(color_image_tensor.shape[1], color_image_tensor.shape[2]) < density
    hints[:, mask] = color_image_tensor[:, mask]
    return hints


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = root_dir
        self.list_files = sorted(os.listdir(self.root_dir))
        self.image_size = image_size
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.color_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self): return len(self.list_files)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.list_files[index])
        img = np.array(Image.open(path).convert("RGB"))
        w = img.shape[1] // 2
        lineart_pil = Image.fromarray(img[:, :w, :]).convert("L")
        color_pil = Image.fromarray(img[:, w:, :])
        lineart_tensor = self.transform(lineart_pil)
        color_tensor = self.color_transform(color_pil)
        hints_tensor = create_color_hints(color_tensor, Config.HINT_DENSITY)
        return lineart_tensor, color_tensor, hints_tensor


# ==============================================================================
# --- SECTION 3: WORKERS POUR THREADING
# ==============================================================================
class WorkerSignals(QObject):
    finished = pyqtSignal();
    error = pyqtSignal(str);
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int);
    plot_updated = pyqtSignal(object);
    samples_updated = pyqtSignal(list)


class PrepareDataWorker(QObject):
    def __init__(self, source_dir, target_dir, image_size, val_split=0.025):
        super().__init__()
        self.signals = WorkerSignals();
        self.source_dir = source_dir;
        self.target_dir = target_dir
        self.image_size = image_size;
        self.val_split = val_split

    def run(self):
        try:
            image_sizes_list = [128, 256, 512, 1024]
            for i, image_size in enumerate(image_sizes_list):
                self.image_size = image_size
                processed_path = os.path.join(self.target_dir, str(self.image_size))
                train_path = os.path.join(processed_path, "train");
                val_path = os.path.join(processed_path, "val")
                os.makedirs(train_path, exist_ok=True);
                os.makedirs(val_path, exist_ok=True)
                files = [f for f in sorted(os.listdir(self.source_dir)) if
                         f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not files: self.signals.error.emit(
                    "Le dossier source est vide."); self.signals.finished.emit(); return
                # random.shuffle(files)
                split_idx = int(len(files) * (1 - self.val_split))
                train_files, val_files = files[:split_idx], files[split_idx:]
                total_files = len(files)

                def process(file_list, out_dir, offset):
                    for i, filename in enumerate(file_list):
                        try:
                            path = os.path.join(self.source_dir, filename)
                            img = cv2.imread(path)
                            if img is None: continue
                            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
                            inv = 255 - gray
                            blur = cv2.GaussianBlur(inv, (21, 21), 0);
                            inv_blur = 255 - blur
                            lineart = cv2.divide(gray, inv_blur, scale=256.0)
                            lineart_bgr = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)
                            concat = np.concatenate([lineart_bgr, img], axis=1)
                            cv2.imwrite(os.path.join(out_dir, filename), concat)
                            progress = int(((i + 1 + offset) / total_files) * 100)
                            self.signals.progress_updated.emit(progress)
                        except Exception:
                            continue

                self.signals.status_updated.emit(f"Préparation des données {self.image_size}x{self.image_size}...")
                process(train_files, train_path, 0)
                process(val_files, val_path, len(train_files))
                num_train = len(os.listdir(train_path));
                num_val = len(os.listdir(val_path))
                self.signals.status_updated.emit(f"Terminé ! {num_train} train, {num_val} val.")
        except Exception as e:
            self.signals.error.emit(f"Erreur: {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


class TrainingWorker(QObject):
    def __init__(self, target_resolution, lr_g, lr_d, num_residuals, epochs_per_stage):
        super().__init__()
        self.signals = WorkerSignals();
        self.target_resolution = target_resolution
        self.lr_g = lr_g;
        self.lr_d = lr_d;
        self.num_residuals = num_residuals
        self.epochs_per_stage = epochs_per_stage

    def run(self):
        try:
            device = Config.DEVICE
            self.signals.status_updated.emit(f"Démarrage... Device: {device}")

            disc = MultiScaleDiscriminator(in_channels=4).to(device)
            gen = Generator(in_channels=4, out_channels=3, num_residuals=self.num_residuals).to(device)

            adversarial_loss = nn.MSELoss();
            L1_LOSS = nn.L1Loss();
            VGG_LOSS = VGGPerceptualLoss(device)
            g_scaler = torch.cuda.amp.GradScaler();
            d_scaler = torch.cuda.amp.GradScaler()
            resolutions = [128, 256, 512, 1024]
            epochs_per_resolution = [20, 40, 60, 80]
            stages = [res for res in resolutions if res <= self.target_resolution]
            total_epochs = len(stages) * self.epochs_per_stage
            epochs_done = 0;
            losses_history = {"d_loss": [], "g_loss": []}

            for stage_idx, res in enumerate(stages):
                self.epochs_per_stage = epochs_per_resolution[stage_idx]
                self.signals.status_updated.emit(f"--- Stage {stage_idx + 1}/{len(stages)}: Résolution {res}x{res} ---")

                opt_disc = optim.Adam(disc.parameters(), lr=self.lr_d, betas=(Config.BETA1, Config.BETA2))
                opt_gen = optim.Adam(gen.parameters(), lr=self.lr_g, betas=(Config.BETA1, Config.BETA2))
                scheduler_g = optim.lr_scheduler.StepLR(opt_gen, step_size=self.epochs_per_stage // 2, gamma=0.5)
                scheduler_d = optim.lr_scheduler.StepLR(opt_disc, step_size=self.epochs_per_stage // 2, gamma=0.5)
                self.signals.status_updated.emit("Optimiseurs réinitialisés pour le nouveau palier.")

                if stage_idx > 0:
                    prev_res = stages[stage_idx - 1]
                    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR,
                                                   f"gen_{prev_res}px_{self.num_residuals}res.pth.tar")
                    if os.path.exists(checkpoint_path):
                        self.signals.status_updated.emit(f"Chargement du checkpoint de {prev_res}px...")
                        gen.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
                    else:
                        self.signals.error.emit(f"Checkpoint manquant pour {prev_res}px."); return

                data_path = os.path.join(Config.PROCESSED_DATA_DIR, str(res))
                train_data_path = os.path.join(data_path, "train")
                if not os.path.exists(train_data_path) or not os.listdir(train_data_path):
                    self.signals.error.emit(f"Données non trouvées pour {res}px.");
                    return

                batch_size = Config.BATCH_SIZES.get(res, 1)
                train_loader = DataLoader(ColorizationDataset(train_data_path, res), batch_size, True,
                                          num_workers=Config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
                val_loader = DataLoader(ColorizationDataset(os.path.join(data_path, "val"), res), 1, False)

                for epoch in range(self.epochs_per_stage):
                    is_hint_phase = (epochs_done + epoch) >= (total_epochs * Config.HINT_PHASE_START_RATIO)
                    if is_hint_phase and epoch == 0 and stage_idx == 0:
                        self.signals.status_updated.emit("--- DÉBUT DE LA PHASE D'ENTRAÎNEMENT AVEC INDICES ---")

                    gen.train();
                    disc.train();
                    epoch_losses = {"d_loss": 0, "g_loss": 0};
                    num_batches = len(train_loader)
                    for idx, (lineart, color, hints) in enumerate(train_loader):
                        lineart, color, hints = lineart.to(device), color.to(device), hints.to(device)
                        real_label = 1.0;
                        fake_label = 0.0

                        if not is_hint_phase: hints = torch.zeros_like(hints)
                        gen_input = torch.cat([lineart, hints], dim=1)

                        with torch.cuda.amp.autocast():
                            y_fake = gen(gen_input);
                            D_real_preds = disc(lineart, color);
                            D_fake_preds = disc(lineart, y_fake.detach());
                            D_loss = 0
                            for D_real, D_fake in zip(D_real_preds, D_fake_preds):
                                D_loss += (adversarial_loss(D_real,
                                                            torch.ones_like(D_real) * real_label) + adversarial_loss(
                                    D_fake, torch.ones_like(D_fake) * fake_label))
                        opt_disc.zero_grad();
                        d_scaler.scale(D_loss).backward();
                        d_scaler.step(opt_disc);
                        d_scaler.update()

                        with torch.cuda.amp.autocast():
                            y_fake_for_g = gen(gen_input);
                            D_fake_preds_for_g = disc(lineart, y_fake_for_g);
                            G_adv_loss = 0
                            for D_fake in D_fake_preds_for_g: G_adv_loss += adversarial_loss(D_fake, torch.ones_like(
                                D_fake) * real_label)
                            G_feat_loss = 0;
                            feat_weights = 4.0 / (3 + 1);
                            D_weights = 1.0 / disc.num_d
                            real_features = disc.forward_features(lineart, color);
                            fake_features = disc.forward_features(lineart, y_fake_for_g)
                            for D_real_feat, D_fake_feat in zip(real_features, fake_features):
                                for i in range(len(D_real_feat) - 1):
                                    G_feat_loss += D_weights * feat_weights * L1_LOSS(D_fake_feat[i], D_real_feat[
                                        i].detach()) * Config.FEATURE_MATCHING_LAMBDA
                            L1 = L1_LOSS(y_fake_for_g, color) * Config.L1_LAMBDA
                            VGG = VGG_LOSS(y_fake_for_g, color) * Config.VGG_LAMBDA
                            G_loss = G_adv_loss + G_feat_loss + L1 + VGG
                        opt_gen.zero_grad();
                        g_scaler.scale(G_loss).backward();
                        g_scaler.step(opt_gen);
                        g_scaler.update()

                        epoch_losses["d_loss"] += D_loss.item();
                        epoch_losses["g_loss"] += G_loss.item()
                        current_epoch_overall = epochs_done + epoch
                        progress = int(((current_epoch_overall * num_batches + idx + 1) / (
                                    total_epochs * num_batches)) * 100) if total_epochs > 0 else 0
                        self.signals.progress_updated.emit(progress)
                        self.signals.status_updated.emit(
                            f"Res: {res}px | Epoch [{epoch + 1}/{self.epochs_per_stage}] | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")

                    scheduler_g.step();
                    scheduler_d.step()
                    losses_history["d_loss"].append(epoch_losses["d_loss"] / num_batches);
                    losses_history["g_loss"].append(epoch_losses["g_loss"] / num_batches)
                    fig = Figure();
                    ax = fig.subplots();
                    ax.plot(losses_history["d_loss"], label="D Loss");
                    ax.plot(losses_history["g_loss"], label="G Loss")
                    ax.legend();
                    ax.grid(True);
                    ax.set_xlabel("Total Epochs");
                    ax.set_ylabel("Loss");
                    fig.tight_layout()
                    self.signals.plot_updated.emit(fig)

                    gen.eval()
                    with torch.no_grad():
                        samples = []
                        for i, (lineart_val, color_val, hints_val) in enumerate(val_loader):
                            if i >= 4: break
                            lineart_val, color_val, hints_val = lineart_val.to(device), color_val.to(
                                device), hints_val.to(device)
                            val_input = torch.cat([lineart_val, hints_val], dim=1)
                            y_fake_val = gen(val_input) * 0.5 + 0.5;
                            lineart_val = lineart_val * 0.5 + 0.5;
                            color_val = color_val * 0.5 + 0.5
                            concat = torch.cat([lineart_val.repeat(1, 3, 1, 1), y_fake_val, color_val], 3).squeeze(0)
                            pil_img = transforms.ToPILImage()(concat)
                            samples.append(pil_img)
                        self.signals.samples_updated.emit(samples)

                epochs_done += self.epochs_per_stage
                torch.save({"state_dict": gen.state_dict()},
                           os.path.join(Config.CHECKPOINT_DIR, f"gen_{res}px_{self.num_residuals}res.pth.tar"))
                self.signals.status_updated.emit(f"Stage {res}px terminé. Checkpoint sauvegardé.")
            self.signals.status_updated.emit("Entraînement progressif terminé !")
        except Exception as e:
            self.signals.error.emit(f"Erreur: {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


# ==============================================================================
# --- SECTION 4: APPLICATION PYQT6
# ==============================================================================
class ImageLabel(QLabel):
    imageChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent);
        self.input_image_path = None;
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.setText("Déposez une image ici ou\ncliquez pour en sélectionner une")
        self.setStyleSheet("ImageLabel { border: 2px dashed #aaa; border-radius: 8px; color: #aaa; }")

    def _set_pix_and_emit(self, path, pix):
        self.input_image_path = path
        self.setPixmap(
            pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.imageChanged.emit(path if path else "")

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasImage() or (mime.hasUrls() and any(
                u.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg')) for u in mime.urls())):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls():
            path = mime.urls()[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg')): self._set_pix_and_emit(path, QPixmap(path))
        elif mime.hasImage():
            self._set_pix_and_emit("", QPixmap.fromImage(mime.imageData()))
        event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", "Images (*.png *.jpg *.jpeg)")
            if path: self._set_pix_and_emit(path, QPixmap(path))
        else:
            super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Application de Colorisation par IA (Inférence Corrigée)");
        self.setGeometry(100, 100, 1200, 800)
        self.thread = None;
        self.worker = None;
        self.input_image_path = None
        self.input_hint_image_path = None
        self.tabs = QTabWidget();
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self.create_data_tab(), "1. Configuration & Données")
        self.tabs.addTab(self.create_training_tab(), "2. Entraînement & Suivi")
        self.tabs.addTab(self.create_inference_tab(), "3. Inférence")

    def create_data_tab(self):
        widget = QWidget();
        layout = QVBoxLayout(widget);
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(QLabel("<h2>Préparation du jeu de données multi-résolution</h2>"))
        layout.addWidget(QLabel(f"Placez vos images couleur dans <b>{Config.RAW_DATA_DIR}</b>."))
        self.prepare_res_combo = QComboBox();
        self.prepare_res_combo.addItems(["128", "256", "512", "1024"])
        self.prepare_button = QPushButton("Lancer la préparation pour la résolution sélectionnée")
        self.prepare_button.clicked.connect(self.run_prepare_data)
        self.prepare_status = QTextEdit();
        self.prepare_status.setReadOnly(True)
        self.prepare_progress = QProgressBar();
        form = QFormLayout()
        form.addRow("Résolution à préparer:", self.prepare_res_combo)
        layout.addLayout(form);
        layout.addWidget(self.prepare_button);
        layout.addWidget(QLabel("Statut:"))
        layout.addWidget(self.prepare_status);
        layout.addWidget(self.prepare_progress)
        return widget

    def create_training_tab(self):
        widget = QWidget();
        main_layout = QHBoxLayout(widget)
        params_widget = QWidget();
        params_layout = QVBoxLayout(params_widget)
        params_layout.setAlignment(Qt.AlignmentFlag.AlignTop);
        params_widget.setFixedWidth(350)
        params_layout.addWidget(QLabel("<h3>Paramètres de l'Entraînement</h3>"))
        form_layout = QFormLayout()
        self.target_res_combo = QComboBox();
        self.target_res_combo.addItems(["128", "256", "512", "1024"]);
        self.target_res_combo.setCurrentText("1024")
        self.epochs_per_stage_spin = QSpinBox();
        self.epochs_per_stage_spin.setRange(10, 200);
        self.epochs_per_stage_spin.setValue(Config.NUM_EPOCHS_PER_STAGE)
        self.num_residuals_spin = QSpinBox();
        self.num_residuals_spin.setRange(3, 20);
        self.num_residuals_spin.setValue(Config.NUM_RESIDUALS)
        self.lr_g_field = QDoubleSpinBox();
        self.lr_g_field.setDecimals(5);
        self.lr_g_field.setSingleStep(1e-05);
        self.lr_g_field.setValue(Config.LEARNING_RATE_G)
        self.lr_d_field = QDoubleSpinBox();
        self.lr_d_field.setDecimals(5);
        self.lr_d_field.setSingleStep(1e-05);
        self.lr_d_field.setValue(Config.LEARNING_RATE_D)
        form_layout.addRow("Résolution Cible Finale:", self.target_res_combo)
        form_layout.addRow("Epochs par Palier:", self.epochs_per_stage_spin)
        form_layout.addRow("Profondeur du Générateur:", self.num_residuals_spin)
        form_layout.addRow("LR Générateur:", self.lr_g_field)
        form_layout.addRow("LR Discriminateur:", self.lr_d_field)
        params_layout.addLayout(form_layout)
        self.start_button = QPushButton("Lancer l'entraînement progressif")
        self.start_button.clicked.connect(self.run_training)
        self.training_status = QLabel("Prêt.")
        self.training_progress = QProgressBar()
        params_layout.addWidget(self.start_button);
        params_layout.addWidget(self.training_status)
        params_layout.addWidget(self.training_progress);
        params_layout.addStretch()
        monitor_widget = QWidget();
        monitor_layout = QVBoxLayout(monitor_widget)
        monitor_layout.addWidget(QLabel("<h3>Suivi en temps réel</h3>"))
        self.loss_canvas = FigureCanvas(Figure());
        self.sample_gallery_layout = QGridLayout()
        monitor_layout.addWidget(QLabel("Courbes de perte:"));
        monitor_layout.addWidget(self.loss_canvas)
        monitor_layout.addWidget(QLabel("Échantillons de validation:"));
        monitor_layout.addLayout(self.sample_gallery_layout);
        monitor_layout.addStretch()
        main_layout.addWidget(params_widget);
        main_layout.addWidget(monitor_widget)
        return widget

    def create_inference_tab(self):
        self.inference_widget = QWidget();
        layout = QGridLayout(self.inference_widget)
        layout.addWidget(QLabel("<h2>Utilisation du modèle</h2>"), 0, 0, 1, 2)
        self.checkpoint_combo = QComboBox();
        self.refresh_checkpoints_btn = QPushButton("Rafraîchir")
        self.refresh_checkpoints_btn.clicked.connect(self.update_checkpoints)
        checkpoint_layout = QHBoxLayout();
        checkpoint_layout.addWidget(self.checkpoint_combo);
        checkpoint_layout.addWidget(self.refresh_checkpoints_btn)
        form_layout = QFormLayout();
        form_layout.addRow("Choisir un checkpoint:", checkpoint_layout)
        self.auto_convert_checkbox = QCheckBox("Auto-convertir l'image couleur en lineart")
        form_layout.addRow(self.auto_convert_checkbox)
        layout.addLayout(form_layout, 1, 0, 1, 2)
        self.input_image_label = ImageLabel();
        self.input_image_label.imageChanged.connect(self.on_image_changed)
        self.input__hint_image_label = ImageLabel();
        self.input__hint_image_label.imageChanged.connect(self.on_hint_image_changed)
        self.output_image_label = QLabel();
        self.output_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_image_label.setText("Le résultat apparaîtra ici");
        self.output_image_label.setStyleSheet("border: 2px solid #333; border-radius: 8px;")
        layout.addWidget(self.input_image_label, 2, 0)
        layout.addWidget(self.input__hint_image_label, 2, 1)
        layout.addWidget(self.output_image_label, 2, 2)

        self.input_image_label.setFixedSize(512, 512)
        self.output_image_label.setFixedSize(512, 512)

        self.colorize_button = QPushButton("Coloriser");
        self.colorize_button.clicked.connect(self.run_inference)
        layout.addWidget(self.colorize_button, 3, 0, 1, 2)
        self.update_checkpoints()
        return self.inference_widget

    def on_image_changed(self, path: str):
        self.input_image_path = path or None
        print("Chemin de l'image mis à jour:", self.input_image_path)

    def on_hint_image_changed(self, path: str):
        self.input_hint_image_path = path or None
        print("Chemin de l'image mis à jour:", self.input_hint_image_path)

    def run_prepare_data(self):
        self.prepare_button.setEnabled(False);
        self.prepare_status.clear();
        self.thread = QThread()
        res = int(self.prepare_res_combo.currentText())
        self.worker = PrepareDataWorker(Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR, res)
        self.worker.moveToThread(self.thread)
        self.worker.signals.status_updated.connect(lambda s: self.prepare_status.append(s))
        self.worker.signals.progress_updated.connect(self.prepare_progress.setValue)
        self.worker.signals.error.connect(self.show_error)
        self.worker.signals.finished.connect(self.thread.quit);
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater);
        self.thread.finished.connect(lambda: self.prepare_button.setEnabled(True))
        self.thread.started.connect(self.worker.run);
        self.thread.start()

    def run_training(self):
        self.start_button.setEnabled(False);
        self.thread = QThread()
        self.worker = TrainingWorker(
            int(self.target_res_combo.currentText()),
            self.lr_g_field.value(),
            self.lr_d_field.value(),
            self.num_residuals_spin.value(),
            self.epochs_per_stage_spin.value()
        )
        self.worker.moveToThread(self.thread)
        self.worker.signals.status_updated.connect(self.training_status.setText)
        self.worker.signals.progress_updated.connect(self.training_progress.setValue)
        self.worker.signals.plot_updated.connect(self.update_plot)
        self.worker.signals.samples_updated.connect(self.update_samples)
        self.worker.signals.error.connect(self.show_error)
        self.worker.signals.finished.connect(self.thread.quit);
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater);
        self.thread.finished.connect(lambda: self.start_button.setEnabled(True))
        self.thread.started.connect(self.worker.run);
        self.thread.start()

    def update_plot(self, fig):
        old_canvas = self.loss_canvas
        self.loss_canvas = FigureCanvas(fig)
        monitor_layout = self.tabs.widget(1).layout().itemAt(1).widget().layout()
        monitor_layout.replaceWidget(old_canvas, self.loss_canvas)
        old_canvas.setParent(None)

    def update_samples(self, samples_pil):
        for i in reversed(range(self.sample_gallery_layout.count())): self.sample_gallery_layout.itemAt(
            i).widget().setParent(None)
        for i, pil_img in enumerate(samples_pil):
            q_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaledToWidth(600, Qt.TransformationMode.SmoothTransformation)
            label = QLabel();
            label.setPixmap(pixmap)
            self.sample_gallery_layout.addWidget(label, i, 0)

    def update_checkpoints(self):
        self.checkpoint_combo.clear()
        if os.path.exists(Config.CHECKPOINT_DIR):
            files = sorted([f for f in os.listdir(Config.CHECKPOINT_DIR) if f.endswith(".pth.tar")], reverse=True)
            self.checkpoint_combo.addItems(files)

    def run_inference(self):
        checkpoint = self.checkpoint_combo.currentText()
        if not checkpoint or not self.input_image_path:
            self.show_error("Veuillez sélectionner un checkpoint et une image.");
            return
        try:
            device = Config.DEVICE

            num_residuals_match = re.search(r'_(\d+)res\.pth\.tar$', checkpoint)
            if num_residuals_match:
                num_residuals = int(num_residuals_match.group(1))
            else:
                num_residuals = Config.NUM_RESIDUALS

            gen = Generator(4, 3, num_residuals=num_residuals).to(device)
            gen.load_state_dict(
                torch.load(os.path.join(Config.CHECKPOINT_DIR, checkpoint), map_location=device)["state_dict"])
            gen.eval()

            # Utiliser une regex plus robuste pour la résolution
            res_match = re.search(r'_(\d+)px_', checkpoint)
            inference_size = int(res_match.group(1)) if res_match else 256

            if self.auto_convert_checkbox.isChecked():
                img_cv = cv2.imread(self.input_image_path)
                if img_cv is None: raise ValueError("Impossible de charger l'image.")
                img_cv = cv2.resize(img_cv, (inference_size, inference_size), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY);
                inv = 255 - gray
                blur = cv2.GaussianBlur(inv, (21, 21), 0);
                inv_blur = 255 - blur
                lineart_np = cv2.divide(gray, inv_blur, scale=256.0)
                lineart_pil = Image.fromarray(lineart_np).convert("L")
                self.input_image_label.setPixmap(QPixmap.fromImage(ImageQt(lineart_pil)).scaled(self.input_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
            else:
                lineart_pil = Image.open(self.input_image_path).convert("L")

            transform = transforms.Compose([transforms.Resize((inference_size, inference_size)), transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])])

            lineart_tensor = transform(lineart_pil).unsqueeze(0).to(device)
            if self.input_hint_image_path:
                hint_pil = Image.open(self.input_hint_image_path).convert("RGB")
                hints_tensor = transform(hint_pil).unsqueeze(0).to(device)
            else:
                hints_tensor = torch.zeros(1, 3, inference_size, inference_size).to(device)
            gen_input = torch.cat([lineart_tensor, hints_tensor], dim=1)

            with torch.no_grad():
                out_tensor = gen(gen_input).squeeze(0).cpu() * 0.5 + 0.5
            out_pil = transforms.ToPILImage()(out_tensor)
            q_img = QImage(out_pil.tobytes(), out_pil.width, out_pil.height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.output_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
            self.output_image_label.setPixmap(pixmap)
        except Exception as e:
            self.show_error(f"Erreur d'inférence: {e}\n{traceback.format_exc()}")

    def show_error(self, message):
        QMessageBox.critical(self, "Erreur", message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt6'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
