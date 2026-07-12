import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PairMaker:
    """
    Splits images from a dataset into complementary pixel-pair tensors
    using disjoint patch sampling.

    Args:
        dataset_name (str): "MNIST" or "CIFAR10"
        root         (str): path to store/load dataset
        patch_size   (int): side length of each square patch (default: 2)
        We use chessboard masking strategy be defualt now. So patch_size is not used.
        batch_size   (int): batch size for DataLoader (default: 100)
    """

    def __init__(self, dataset_name="MNIST", root="./data", patch_size=2, batch_size=100):
        self.dataset_name = dataset_name
        self.patch_size   = patch_size
        # Safe defaults so every dataset has these attributes set. These are
        # tuned for 28x28 (MNIST / FashionMNIST).
        self.rot_val      = 20.0
        self.rot_prob     = 1.0
        self.flip_prob    = 0.5
        self.brightness   = 0.3
        self.contrast     = 0.3
        self.saturation   = 0.3
        self.min_edge     = 24
        # After chessboard masking, optionally upsample EACH view to a common
        # working resolution. USPS is masked at its native 16x16 (so the two
        # views never share interpolated pixels — no information leakage) and
        # only THEN is each view independently upsampled to 32x32. MNIST /
        # FashionMNIST / CIFAR10 keep their native resolution (None = no resize).
        self.resize_after_mask = None
        # Crop augmentation: with prob crop_prob, an H_c×W_c region (H_c, W_c
        # sampled independently in {min_edge..S}) is cropped at a random position
        # and resized back to S×S. Otherwise the full S×S image is kept (H_c=W_c=S)
        # and only the zoom-out step (below) may apply.
        self.crop_prob       = 1.0
        # Zoom-out augmentation: with prob zoomout_prob, the crop is padded with a
        # black border before being resized back, shrinking the content. The LONGER
        # crop edge is padded — pad (H_c+L)×W_c if H_c>=W_c else H_c×(W_c+L), with
        # L ~ Uniform{zoomout_min_pad..zoomout_max_pad}. Padding the longer edge
        # amplifies the crop's own aspect distortion (the already-stretched axis is
        # compressed further), strengthening the tall/short, wide/narrow stretch.
        self.zoomout_prob    = 0.0
        self.zoomout_min_pad = 2
        self.zoomout_max_pad = 6
        # Brightness match: after masking (+ resize), bilinear upsampling averages
        # each visible pixel with its zeroed chessboard neighbors, which roughly
        # halves stroke brightness for resized datasets (USPS). When True, each
        # view is rescaled so its p99 intensity matches the source image's p99 —
        # removing the systematic dimming so the brightness/contrast jitter then
        # varies symmetrically around the original brightness. OFF by default —
        # only USPS resizes and thus needs it; MNIST/Fashion keep stroke brightness
        # already (factor ≈ 1), so we leave it off there to exactly match the
        # original pipeline.
        self.match_brightness = False
        if dataset_name == 'USPS':
            print("USPS: mask at native 16x16, upsample each view to 32x32, milder augmentation")
            self.rot_val      = 10
            self.rot_prob     = 0.5
            self.brightness   = 0.15
            self.contrast     = 0.15
            self.min_edge          = 30
            self.resize_after_mask = 32
            self.zoomout_prob    = 1.0
            self.zoomout_min_pad = 2
            self.zoomout_max_pad = 6
            self.match_brightness = True
        self._to_pil      = transforms.ToPILImage()
        self._to_tensor   = transforms.ToTensor()

        transform = transforms.Compose([transforms.ToTensor()])

        if dataset_name == "MNIST":
            train_dataset = datasets.MNIST(root=root, train=True,  download=True, transform=transform)
            test_dataset  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        elif dataset_name == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(root=root, train=True,  download=True, transform=transform)
            test_dataset  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        elif dataset_name == "CIFAR10":
            train_dataset = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform)
            test_dataset  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        elif dataset_name == "USPS":
            train_dataset = datasets.USPS(root=root, train=True,  download=True, transform=transform)
            test_dataset  = datasets.USPS(root=root, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


    def _augment_batch(
        self,
        x: torch.Tensor,
        rot_val=None,
        rot_prob=None,
        min_edge=None,
        crop_prob=None,
        flip_prob=None,
        brightness=None,
        contrast=None,
        saturation=None,
        zoomout_prob=None,
        zoomout_min_pad=None,
        zoomout_max_pad=None,
    ) -> torch.Tensor:
        """
        GPU-native batched augmentation replacing the PIL per-image loop.
        x: (N, C, H, W) float tensor on any device (H == W assumed).
        Returns augmented (N, C, H, W) on same device.

        Augmentation order: (1) anisotropic crop + optional zoom-out, then
        (2) rotation, (3) horizontal flip, (4) brightness/contrast/saturation.

        Args:
            rot_val      : max rotation degrees (±). None → use self.rot_val.
            rot_prob     : probability that any given image is rotated. Set 0
                           to disable rotation entirely.
            min_edge     : minimum crop edge length in pixels. Per image, the
                           vertical edge H_c and horizontal edge W_c are each
                           sampled INDEPENDENTLY and uniformly from
                           {min_edge, ..., S} (S = image side). A H_c×W_c region
                           is cropped at a random position and resized back to
                           S×S — independent H_c/W_c decorrelate the aspect
                           ratio between the two views (tall/short, wide/narrow).
                           min_edge >= S disables cropping. Default 24.
            crop_prob    : probability that cropping is applied per image. If it
                           does not fire, the full S×S image is kept (H_c=W_c=S)
                           and only the zoom-out step may apply. None → self.crop_prob.
            flip_prob    : probability of horizontal flip (non-digit only).
                           Set 0 to disable.
            brightness   : ± multiplicative jitter. 0 → disabled.
            contrast     : ± multiplicative jitter on per-image-mean deviation. 0 → disabled.
            saturation   : ± multiplicative jitter (RGB only). 0 → disabled.
            zoomout_prob    : probability that the crop is zoomed OUT (the longer
                              crop edge padded with a black border before resizing
                              back), shrinking the content. None → self.zoomout_prob.
            zoomout_min_pad : min black padding L (px) when a zoom-out fires.
                              None → self.zoomout_min_pad.
            zoomout_max_pad : max black padding L (px) when a zoom-out fires;
                              L ~ Uniform{zoomout_min_pad..zoomout_max_pad}.
                              None → self.zoomout_max_pad.
        """
        N, C, H, W = x.shape
        device = x.device

        if rot_val is None:
            rot_val = self.rot_val
        if rot_prob is None:
            rot_prob = self.rot_prob
        if min_edge is None:
            min_edge = self.min_edge
        if crop_prob is None:
            crop_prob = self.crop_prob
        if flip_prob is None:
            flip_prob = self.flip_prob
        if brightness is None:
            brightness = self.brightness
        if contrast is None:
            contrast = self.contrast
        if saturation is None:
            saturation = self.saturation
        if zoomout_prob is None:
            zoomout_prob = self.zoomout_prob
        if zoomout_min_pad is None:
            zoomout_min_pad = self.zoomout_min_pad
        if zoomout_max_pad is None:
            zoomout_max_pad = self.zoomout_max_pad

        # ── 1. Random (square) crop → single-axis black pad (zoom-out) → resize,
        #      in ONE interpolation ──────────────────────────────────────────────
        # Per image (S == H == W):
        #   (1) with prob crop_prob, sample ONE edge E ~ Uniform{min_edge..S} and
        #       use it for BOTH axes (E×E, isotropic — aspect ratio preserved, same
        #       as the original pipeline) at a random position (r0, c0); otherwise
        #       keep the full image (E = S);
        #   (2) with prob zoomout_prob, pad ONE randomly chosen crop edge with
        #       black: (E+L)×E or E×(E+L), L ~ Uniform{zoomout_min_pad..zoomout_max_pad},
        #       crop placed at a random offset in the black canvas;
        #   (3) resize the canvas back to S×S.
        # The crop itself no longer distorts aspect (square); any tall/short,
        # wide/narrow stretch comes solely from the single-axis zoom-out pad in (2).
        # Steps (1)-(3) compose into a single affine (crop+resize geometry) + a
        # multiplicative black mask (the pad), so the whole thing costs ONE
        # grid_sample / one interpolation — no double resize.
        S = H
        can_crop    = min_edge < S and crop_prob > 0
        do_zoomout  = zoomout_max_pad > 0 and zoomout_prob > 0
        if can_crop or do_zoomout:
            # (1) crop edge + position (per-image crop gate). ISOTROPIC crop: one
            # edge E is sampled and used for BOTH axes (edge_h == edge_w == E), so
            # the crop is square and preserves the aspect ratio — identical to the
            # original pipeline.
            edge_h = torch.full((N,), float(S), device=device)
            edge_w = torch.full((N,), float(S), device=device)
            if can_crop:
                n_choices = S - min_edge + 1
                crop_mask = (torch.rand(N, device=device) < crop_prob)
                samp_e = (min_edge + torch.randint(0, n_choices, (N,), device=device)).to(torch.float32)
                edge_h = torch.where(crop_mask, samp_e, edge_h)
                edge_w = torch.where(crop_mask, samp_e, edge_w)   # same E -> square crop
            r0 = torch.rand(N, device=device) * (float(S) - edge_h)   # top  ∈ [0, S-E]
            c0 = torch.rand(N, device=device) * (float(S) - edge_w)   # left ∈ [0, S-E]

            # (2) single-axis black pad (zoom-out): pad ONE randomly chosen crop
            # edge by L. The crop is now square (edge_h == edge_w), so there is no
            # "longer" edge to prefer — the axis is chosen at random. Padding only
            # one axis makes the zoom-out anisotropic, giving the tall/short,
            # wide/narrow stretch.
            pad_h = torch.zeros(N, device=device)
            pad_w = torch.zeros(N, device=device)
            if do_zoomout:
                do_zoom = torch.rand(N, device=device) < zoomout_prob
                pick_h  = torch.rand(N, device=device) < 0.5                       # pad a random axis
                L = (zoomout_min_pad + torch.randint(
                        0, int(zoomout_max_pad) - int(zoomout_min_pad) + 1, (N,), device=device)
                    ).to(torch.float32)
                pad_h = torch.where(do_zoom & pick_h,  L, pad_h)
                pad_w = torch.where(do_zoom & ~pick_h, L, pad_w)
            canvas_h = edge_h + pad_h        # (N,) padded-canvas height
            canvas_w = edge_w + pad_w        # (N,) padded-canvas width
            p_top  = torch.rand(N, device=device) * pad_h   # crop offset in canvas ∈ [0, L1]
            p_left = torch.rand(N, device=device) * pad_w   # crop offset in canvas ∈ [0, L2]

            # (3) one affine: output (S×S) ← padded canvas, mapped onto the input
            #     crop. Convention: output samples the full canvas in normalized
            #     [-1,1]; the crop occupies canvas rows/cols [p_*, p_*+edge_*],
            #     mapped to input rows/cols [r0/c0, r0/c0+edge_*]. Matching those
            #     two edges fixes the per-axis scale a = canvas/S and shift t.
            a_x = canvas_w / float(S)
            a_y = canvas_h / float(S)
            # crop edges in input-normalized coords
            s_left = 2.0 * c0 / float(S) - 1.0
            s_top  = 2.0 * r0 / float(S) - 1.0
            # crop edges in output/canvas-normalized coords
            g_left = 2.0 * p_left / canvas_w - 1.0
            g_top  = 2.0 * p_top  / canvas_h - 1.0
            tx = s_left - a_x * g_left
            ty = s_top  - a_y * g_top

            zeros = torch.zeros(N, device=device)
            theta = torch.stack([
                a_x,   zeros, tx,
                zeros, a_y,   ty
            ], dim=1).reshape(N, 2, 3)
            grid = F.affine_grid(theta, (N, C, S, S), align_corners=False)
            # padding_mode='border' so a pure crop (no pad) never leaks black;
            # the black pad border is applied explicitly by the mask below.
            x = F.grid_sample(x, grid, mode='nearest', padding_mode='border', align_corners=False)

            # Black-pad mask: an output pixel is content iff its (canvas-space)
            # center falls inside the crop rectangle [g_left, g_right] ×
            # [g_top, g_bottom]; otherwise it is padding → black. No interpolation.
            if do_zoomout:
                g_right  = g_left + 2.0 * edge_w / canvas_w
                g_bottom = g_top  + 2.0 * edge_h / canvas_h
                centers  = 2.0 * (torch.arange(S, device=device).to(torch.float32) + 0.5) / float(S) - 1.0
                in_x = (centers[None, :] >= g_left[:, None]) & (centers[None, :] <= g_right[:, None])   # (N,S)
                in_y = (centers[None, :] >= g_top[:, None])  & (centers[None, :] <= g_bottom[:, None])  # (N,S)
                mask = (in_y[:, :, None] & in_x[:, None, :]).to(x.dtype)                                # (N,S,S)
                x = x * mask.unsqueeze(1)

        # ── 2. Random rotation (rot_prob chance, ±rot_val degrees) ─────────────
        if rot_val != 0 and rot_prob > 0:
            angles = torch.where(
                torch.rand(N, device=device) < rot_prob,
                torch.FloatTensor(N).uniform_(-rot_val, rot_val).to(device),
                torch.zeros(N, device=device)
            )
            cos_a = torch.cos(angles * math.pi / 180.)
            sin_a = torch.sin(angles * math.pi / 180.)
            zeros = torch.zeros(N, device=device)
            # Affine matrix rows: [[cos, -sin, 0], [sin, cos, 0]]
            theta_rot = torch.stack([
                cos_a, -sin_a, zeros,
                sin_a,  cos_a, zeros
            ], dim=1).reshape(N, 2, 3)
            grid = F.affine_grid(theta_rot, (N, C, H, W), align_corners=False)
            x = F.grid_sample(x, grid, mode='nearest', padding_mode='zeros', align_corners=False)

        # ── 3. Horizontal flip (skip for digit datasets) ───────────────────────
        # MNIST and USPS are handwritten digits — a mirrored digit is not the
        # same class, so flipping is disabled. FashionMNIST / CIFAR do flip.
        if self.dataset_name not in ("MNIST", "USPS") and flip_prob > 0:
            flip_mask = torch.rand(N, device=device) < flip_prob  # (N,)
            x[flip_mask] = x[flip_mask].flip(-1)

        # ── 4. ColorJitter (brightness, contrast, saturation) ──────────────────
        # brightness: multiply by U[1-brightness, 1+brightness]
        if brightness > 0:
            b = torch.FloatTensor(N, 1, 1, 1).uniform_(1 - brightness, 1 + brightness).to(device)
            x = (x * b).clamp(0, 1)

        # contrast: per-image mean, then scale deviation
        if contrast > 0:
            c = torch.FloatTensor(N, 1, 1, 1).uniform_(1 - contrast, 1 + contrast).to(device)
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = ((x - mean) * c + mean).clamp(0, 1)

        # saturation: only meaningful for RGB; skip for grayscale (C==1)
        if C == 3 and saturation > 0:
            s = torch.FloatTensor(N, 1, 1, 1).uniform_(1 - saturation, 1 + saturation).to(device)
            gray = x.mean(dim=1, keepdim=True).expand_as(x)
            x = (gray + (x - gray) * s).clamp(0, 1)

        # hue: skip (requires HSV conversion, expensive; minor effect)

        return x


    # ── Internal helper ───────────────────────────────────────────────────────
    def _smooth_chessboard(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Fill masked-out positions with the mean of their direct (up/down/
        left/right) neighbors. Relies on the chessboard property: every
        masked position's 4 in-bounds direct neighbors are visible
        (opposite parity). At image edges, only the in-bounds neighbors
        contribute — averaged over however many of them there are (2 or 3).

        Args:
            tensor : (N, C, H, W)  masked image (0 at masked positions).
            mask   : (N, 1, H, W)  1 where visible, 0 where masked.

        Returns:
            (N, C, H, W)  same shape, masked positions replaced with the
            neighbor mean; visible positions kept intact.
        """
        N, C, H, W = tensor.shape
        device = tensor.device
        dtype  = tensor.dtype

        # 3x3 cross kernel: averages over up/down/left/right neighbors only.
        cross = torch.tensor(
            [[0., 1., 0.],
             [1., 0., 1.],
             [0., 1., 0.]],
            dtype=dtype, device=device,
        ).view(1, 1, 3, 3)

        # Per-channel neighbor sum (groups=C → C independent depthwise convs).
        kernel_c = cross.expand(C, 1, 3, 3)
        neighbor_sum = F.conv2d(tensor, kernel_c, padding=1, groups=C)

        # Count of visible in-bounds neighbors per position (single-channel
        # since mask is (N, 1, H, W)).
        neighbor_count = F.conv2d(mask, cross, padding=1)
        neighbor_count_safe = neighbor_count.clamp(min=1.0)

        neighbor_mean = neighbor_sum / neighbor_count_safe

        # Visible positions: keep original. Masked positions: fill with mean.
        return tensor * mask + neighbor_mean * (1.0 - mask)

    def _make_pairs(
        self,
        images,
        augment=2,
        device=None,
        smooth=False,
        rot_val=None,
        rot_prob=None,
        min_edge=None,
        crop_prob=None,
        flip_prob=None,
        brightness=None,
        contrast=None,
        saturation=None,
        zoomout_prob=None,
        zoomout_min_pad=None,
        zoomout_max_pad=None,
    ):
        """
        Chessboard pixel split.

        "Black" positions are (r, c) where (r + c) is even — i.e.
        (0,0), (0,2), (0,4), ..., (1,1), (1,3), (1,5), ....
        "White" positions are the complement.

        For each image in the batch, flip a fair coin:
          - 50%: i-side sees BLACK positions, j-side sees WHITE.
          - 50%: i-side sees WHITE positions, j-side sees BLACK.

        Either side sees exactly H*W/2 pixels (assuming H*W is even, which
        holds for both MNIST 28×28 and CIFAR 32×32). Masked-out pixels are
        set to 0.

        Optional neighbor smoothing fills masked positions with the mean of
        their 4 visible neighbors (controlled by `smooth`, default False).
        Augmentation (rotation/crop/jitter via _augment_batch) is applied
        AFTER the mask + smoothing.

        Augmentation knobs (rot_val, rot_prob, min_edge, flip_prob,
        brightness, contrast, saturation) are forwarded to _augment_batch
        and default to the original aggressive values. Pass narrower values
        in to use a milder policy for a specific call (e.g. a targeted
        cluster batch).
        """
        if device is None:
            device = images.device
        images = images.to(device)

        N, C, H, W = images.shape

        # Build the chessboard pattern once: black_mask[r, c] = 1 iff (r+c)%2==0.
        rows = torch.arange(H, device=device).view(H, 1)
        cols = torch.arange(W, device=device).view(1, W)
        black_mask = ((rows + cols) % 2 == 0).to(images.dtype)        # (H, W)
        black_mask = black_mask.view(1, 1, H, W)                      # broadcast over (N, C)
        white_mask = 1.0 - black_mask                                 # (1, 1, H, W)

        # Per-image fair coin: 1 → i sees black; 0 → i sees white.
        coin = (torch.rand(N, device=device) < 0.5).to(images.dtype)  # (N,)
        coin = coin.view(N, 1, 1, 1)                                  # broadcast

        mask_i = coin * black_mask + (1.0 - coin) * white_mask        # (N, 1, H, W)
        mask_j = 1.0 - mask_i                                         # complementary

        tensor_i = images * mask_i   # broadcasts over the C dim
        tensor_j = images * mask_j

        # ── Optional neighbor smoothing — fill each masked position with the
        # mean of its 4 direct (in-bounds, visible) neighbors. Runs BEFORE
        # augmentation, per the design. Toggle via the `smooth` kwarg.
        if smooth:
            tensor_i = self._smooth_chessboard(tensor_i, mask_i)
            tensor_j = self._smooth_chessboard(tensor_j, mask_j)

        # ── Per-view upsample to a common working resolution (USPS: 16 -> 32). ──
        # Done AFTER masking (and smoothing) so the two views never share
        # interpolated pixels — view_i is a function only of its own visible
        # pixels (+ zeros), so there is no information leakage to view_j. Each
        # view is resized independently. None → no resize (native resolution).
        if self.resize_after_mask is not None:
            R = self.resize_after_mask
            tensor_i = F.interpolate(tensor_i, size=(R, R), mode='bilinear', align_corners=False)
            tensor_j = F.interpolate(tensor_j, size=(R, R), mode='bilinear', align_corners=False)

        # ── Brightness match — undo the resize dimming ──────────────────────────
        # Bilinear upsampling of a half-zeroed (chessboard) image averages visible
        # pixels with zeros, ~halving stroke brightness. Rescale each view so its
        # p99 intensity matches the source image's p99 (per image), so neither
        # view is systematically darker than the original. Done BEFORE augmentation
        # so the brightness/contrast jitter varies symmetrically around it. p99
        # (not max) is used for robustness to single hot pixels.
        if self.match_brightness:
            eps = 1e-4
            q = 0.99
            src_q = torch.quantile(images.reshape(N, -1).float(), q, dim=1).view(N, 1, 1, 1)
            qi = torch.quantile(tensor_i.reshape(N, -1).float(), q, dim=1).view(N, 1, 1, 1)
            qj = torch.quantile(tensor_j.reshape(N, -1).float(), q, dim=1).view(N, 1, 1, 1)
            tensor_i = (tensor_i * (src_q / qi.clamp(min=eps))).clamp(0, 1)
            tensor_j = (tensor_j * (src_q / qj.clamp(min=eps))).clamp(0, 1)

        # ── IIC-style augmentation via PIL (per-image, independent) on both side ──────────
        # ── GPU-native batched augmentation ──────────────────────────────────
        aug_kwargs = dict(
            rot_val=rot_val,
            rot_prob=rot_prob,
            min_edge=min_edge,
            crop_prob=crop_prob,
            flip_prob=flip_prob,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            zoomout_prob=zoomout_prob,
            zoomout_min_pad=zoomout_min_pad,
            zoomout_max_pad=zoomout_max_pad,
        )
        if augment == 2:
            tensor_i = self._augment_batch(tensor_i, **aug_kwargs)
            tensor_j = self._augment_batch(tensor_j, **aug_kwargs)
        elif augment == 1:
            tensor_j = self._augment_batch(tensor_j, **aug_kwargs)

        return tensor_i.to(device), tensor_j.to(device)

    # ── Public API ────────────────────────────────────────────────────────────
    def split_patches(self, N, seed=None, split="train"):
        """
        Randomly selects N images and splits each into complementary pixel pairs.

        seed controls image selection ONLY — patch pixel selection is always random.

        Args:
            N     (int):  number of images to sample
            seed  (int):  optional seed for image selection reproducibility
            split (str):  "train" or "test"

        Returns:
            tensor_i : (N, C, H, W) — selected pixels visible, rest black
            tensor_j : (N, C, H, W) — complementary pixels visible, rest black
            labels   : (N,)         — ground truth labels
        """
        dataset = self.train_dataset if split == "train" else self.test_dataset

        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            indices = torch.randperm(len(dataset), generator=rng)[:N]
        else:
            indices = torch.randperm(len(dataset))[:N]

        images = torch.stack([dataset[i][0] for i in indices])
        labels = torch.tensor([dataset[i][1] for i in indices])

        tensor_i, tensor_j = self._make_pairs(images)
        return tensor_i, tensor_j, labels

    def make_session_loaders(self, R=0.7, seed=None):
        """
        Randomly splits self.train_dataset into two disjoint subsets:
        - partial_train_loader : R portion, used for student training
        - partial_eval_loader  : (1-R) portion, used for surprise score evaluation

        Call this at the beginning of each session.

        Args:
            R    (float): fraction of training data to use for training (default: 0.7)
            seed (int):   optional seed for reproducibility
        """
        N = len(self.train_dataset)
        n_train = int(N * R)
        n_eval  = N - n_train

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        train_subset, eval_subset = torch.utils.data.random_split(
            self.train_dataset,
            [n_train, n_eval],
            generator=generator
        )

        self.partial_train_loader = DataLoader(
            train_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )
        self.partial_eval_loader = DataLoader(
            eval_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=False
        )

    def split_patches_from_loader(self, split="train", device=None, force_augment=None, smooth=False):
        """
        Yields (tensor_i, tensor_j, labels) for each batch from the DataLoader.
        Useful for iterating over the full dataset during training.

        Args:
            split (str): "local_search", "train", "valid" or "test"

        Yields:
            tensor_i : (B, C, H, W)
            tensor_j : (B, C, H, W)
            labels   : (B,)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if split == 'local_search':
            loader = self.train_loader
            augment = 2
        elif split == "train":
            loader = self.partial_train_loader
            augment = 2
        elif split in ("valid", "eval", "val"):
            loader = self.partial_eval_loader
            augment = 2
        elif split in ("test", "testing"):
            loader = self.test_loader
            augment = 0
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if force_augment is not None:
            augment = force_augment

        for images, labels in loader:
            tensor_i, tensor_j = self._make_pairs(images, augment=augment, device=device, smooth=smooth)
            yield tensor_i, tensor_j, labels, images.to(device)

    def split_patch_demo(self, N=4, seed=None, split="train"):
        """
        Runs split_patches and visualises Original / Image_0 / Image_1
        for each sampled image.
        """
        dataset = self.train_dataset if split == "train" else self.test_dataset

        tensor_i, tensor_j, labels = self.split_patches(N, seed=seed, split=split)

        # Reload originals with same seed for side-by-side comparison
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            indices = torch.randperm(len(dataset), generator=rng)[:N]
        else:
            indices = torch.randperm(len(dataset))[:N]
        originals = torch.stack([dataset[i][0] for i in indices])

        print(f"Split          : {split}")
        print(f"tensor_i shape : {tensor_i.shape}")
        print(f"tensor_j shape : {tensor_j.shape}")
        print(f"labels         : {labels.tolist()}")

        fig, axes = plt.subplots(N, 3, figsize=(7, N * 2.2))
        col_titles = ["Original", "Image_0 (selected)", "Image_1 (complement)"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10)

        for n in range(N):
            for col, img in enumerate([originals[n], tensor_i[n], tensor_j[n]]):
                axes[n, col].imshow(img.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
                axes[n, col].axis("off")
            axes[n, 0].set_ylabel(f"label={labels[n].item()}", fontsize=9)

        plt.suptitle(f"{self.dataset_name} — patch_size={self.patch_size} ({split})", fontsize=11)
        plt.tight_layout()
        plt.show()

        return tensor_i, tensor_j, labels


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pm = PairMaker(dataset_name="MNIST")

    # Demo: visualise 4 training images
    pm.split_patch_demo(N=4, seed=42, split="train")

    # Example: iterate full training set batch by batch
    for tensor_i, tensor_j, labels, images in pm.split_patches_from_loader(split="train"):
        print(f"batch — i: {tensor_i.shape}, j: {tensor_j.shape}, labels: {labels.shape}, images: {images.shape}")
        break  # remove to iterate full dataset
