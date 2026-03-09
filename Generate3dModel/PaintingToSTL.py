import os
import math

import cv2
import numpy as np
from PIL import Image
import torch
from stl import mesh
from transformers import pipeline as hf_pipeline
from skimage.restoration import denoise_tv_chambolle

## Parameters

# Point to the image that you want to process
imgName = 'D:\Documents\Reliefeelable\Images\Bright_Unity.jpg'

# Set to true to auto resize the image to target pixel count. Else, will use the resolution of the original image.
autoResize = True

# The pixel count of the image that you want to generate.
# 75000 is a good value for the BambuLab P1S 3D printer.
targetPixelCount = 300000

# Set to true to create a composite image with 4 tiles of the original image.
createComposite = False

# Bilateral filter parameters -- smooths brushstroke noise while preserving object edges
bilateralD = 9
bilateralSigmaColor = 75
bilateralSigmaSpace = 75

# Canny edge detection parameters
cannyThreshold1 = 120
cannyThreshold2 = 240
cannyGaussianBlur = 3
cannyBlendWeight = 0.35   # weight of Canny edges in composite (0 = depth only, 1 = edges only)

# Total Variation denoising strength (higher = smoother surface). Range 0.05 - 0.15 recommended.
tvWeight = 0.1

# STL export parameters
maxDepthMM = 10          # maximum relief height in mm
baseThicknessMM = 0.5   # minimum base plate thickness in mm
pixelSizeMM = 0.66      # physical width/height of one pixel in mm
stlSubsample = 4        # use every Nth pixel for mesh (4 = 16x fewer vertices; simpler STL)
maxPrintSizeMM = 230    # scale STL so footprint fits within this size (mm); e.g. 230 = 23cm for 25cm bed

# Set to True to show intermediate images and pause for user confirmation between steps.
interactive = True

## Auto-detect device

device = "cuda" if torch.cuda.is_available() else "cpu"

## Processing

fileName = os.path.splitext(os.path.basename(imgName))[0]

if not os.path.exists('out'):
    os.mkdir('out')

folder = os.path.join('out', fileName)
if not os.path.exists(folder):
    os.mkdir(folder)

### Helper functions

def show_and_confirm(title, images):
    """
    Display one or more images side-by-side and wait for user input.
    images: list of (label, img) tuples. img can be a BGR uint8 array
            or a float [0,1] depth map (auto-converted to grayscale uint8).
    The user presses Enter to continue or types 'q' to abort the pipeline.
    Does nothing if interactive mode is disabled.
    """
    if not interactive:
        return

    display_parts = []
    for label, img in images:
        if img.dtype != np.uint8:
            vis = (img * 255.0).clip(0, 255).astype('uint8')
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        else:
            vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        label_bar = np.zeros((30, vis.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        display_parts.append(np.vstack([label_bar, vis]))

    max_h = max(p.shape[0] for p in display_parts)
    padded = []
    for p in display_parts:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    combined = np.hstack(padded)

    max_display_w = 1600
    if combined.shape[1] > max_display_w:
        scale = max_display_w / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    try:
        cv2.imshow(title, combined)
        cv2.waitKey(1)
    except cv2.error:
        # OpenCV built without GUI (e.g. opencv-python-headless); save preview to output folder
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title).strip().replace(" ", "_")
        preview_path = os.path.join(folder, f"preview_{safe_name}.jpeg")
        cv2.imwrite(preview_path, combined)
        print(f"(Display not available; preview saved to {preview_path})")

    response = input(f"\n[{title}] Press Enter to continue, or type 'q' to quit: ").strip().lower()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    if response == 'q':
        print("Pipeline aborted by user.")
        raise SystemExit(0)


def resize(img, scale_ratio):
    """Resize a cv2 image by scale_ratio, preserving aspect ratio."""
    if not autoResize:
        return img
    width = int(img.shape[1] * scale_ratio)
    height = int(img.shape[0] * scale_ratio)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def resize_pil(pil_img, scale_ratio):
    """Resize a PIL image by scale_ratio, preserving aspect ratio."""
    if not autoResize:
        return pil_img
    width = int(pil_img.width * scale_ratio)
    height = int(pil_img.height * scale_ratio)
    return pil_img.resize((width, height), Image.LANCZOS)


def normalize(depth_map):
    """Normalize a depth map to [0, 1] range."""
    mn, mx = depth_map.min(), depth_map.max()
    return (depth_map - mn) / (mx - mn + 1e-8)


def save_depth_as_image(depth_map, filepath):
    """Save a [0,1]-range depth map as a grayscale JPEG."""
    formatted = (depth_map * 255.0).clip(0, 255).astype('uint8')
    cv2.imwrite(filepath, formatted)


def save_thermal(depth_map, filepath):
    """Save a [0,1]-range depth map with INFERNO colormap."""
    gray = (depth_map * 255.0).clip(0, 255).astype('uint8')
    colorized = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite(filepath, colorized)


def run_canny_edges(bgr_img, threshold1, threshold2, blur_size):
    """Canny edge detection. Returns float64 [0,1] edge map (edges = 1)."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(blurred, threshold1=threshold1, threshold2=threshold2)
    return (edges.astype(np.float64) / 255.0)


def run_depth_anything_v2(pil_img):
    """Run Depth Anything V2 depth estimation. Returns a float64 numpy array (H x W)."""
    print("Loading Depth Anything V2 model...")
    da_pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0 if device == "cuda" else -1,
    )

    print("Running Depth Anything V2 depth estimation...")
    result = da_pipe(pil_img)
    depth = np.array(result["depth"], dtype=np.float64)

    del da_pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return depth


def merge_depth_and_canny(depth_norm, canny_norm, canny_weight):
    """
    Composite depth map with Canny edges. Both in [0,1].
    Returns normalized [0,1] composite (edges add relief).
    """
    composite = (1.0 - canny_weight) * depth_norm + canny_weight * canny_norm
    return normalize(composite)


def save_heightmap_stl(depth_map, filepath, pixel_size_mm, max_height_mm, base_thickness_mm, subsample=1, max_print_size_mm=None):
    """
    Convert a [0,1]-range 2D depth map into an STL mesh using a shared-vertex
    heightmap grid. subsample > 1 reduces vertices (e.g. 4 = 16x fewer triangles).
    """
    if subsample > 1:
        depth_map = depth_map[::subsample, ::subsample].copy()
        pixel_size_mm = pixel_size_mm * subsample
    h, w = depth_map.shape
    print(f"Generating STL mesh ({w}x{h} = {w*h} vertices)...")

    top_verts = np.zeros((h, w, 3), dtype=np.float64)
    bot_verts = np.zeros((h, w, 3), dtype=np.float64)

    xs = np.arange(w, dtype=np.float64) * pixel_size_mm
    ys = np.arange(h, dtype=np.float64) * pixel_size_mm
    xgrid, ygrid = np.meshgrid(xs, ys)

    top_verts[:, :, 0] = xgrid
    top_verts[:, :, 1] = ygrid
    top_verts[:, :, 2] = depth_map * max_height_mm + base_thickness_mm

    bot_verts[:, :, 0] = xgrid
    bot_verts[:, :, 1] = ygrid
    bot_verts[:, :, 2] = 0.0

    top_flat = top_verts.reshape(-1, 3)
    bot_flat = bot_verts.reshape(-1, 3)
    all_verts = np.vstack([top_flat, bot_flat])

    # Scale x,y to fit within max print bed size (e.g. 23cm)
    if max_print_size_mm is not None and max_print_size_mm > 0:
        current_x_mm = (w - 1) * pixel_size_mm
        current_y_mm = (h - 1) * pixel_size_mm
        scale = min(1.0, max_print_size_mm / (current_x_mm + 1e-9), max_print_size_mm / (current_y_mm + 1e-9))
        all_verts[:, 0] *= scale
        all_verts[:, 1] *= scale
        if scale < 1.0:
            print(f"Scaled mesh to fit {max_print_size_mm}mm bed: {current_x_mm:.0f}x{current_y_mm:.0f}mm -> {current_x_mm*scale:.0f}x{current_y_mm*scale:.0f}mm")

    # Mirror left-to-right so the print matches the image orientation
    x_max = all_verts[:, 0].max()
    all_verts[:, 0] = x_max - all_verts[:, 0]

    n_top = h * w  # vertex offset for bottom layer

    faces = []

    # Top surface: two triangles per grid cell
    for j in range(h - 1):
        for i in range(w - 1):
            v0 = j * w + i
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # Bottom surface (winding reversed so normals face down)
    for j in range(h - 1):
        for i in range(w - 1):
            v0 = n_top + j * w + i
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    # Side walls -- four edges of the grid
    # Front edge (j=0)
    for i in range(w - 1):
        t = i
        t1 = i + 1
        b = n_top + i
        b1 = n_top + i + 1
        faces.append([t, t1, b])
        faces.append([t1, b1, b])

    # Back edge (j=h-1)
    for i in range(w - 1):
        t = (h - 1) * w + i
        t1 = t + 1
        b = n_top + (h - 1) * w + i
        b1 = b + 1
        faces.append([t, b, t1])
        faces.append([t1, b, b1])

    # Left edge (i=0)
    for j in range(h - 1):
        t = j * w
        t1 = (j + 1) * w
        b = n_top + j * w
        b1 = n_top + (j + 1) * w
        faces.append([t, b, t1])
        faces.append([t1, b, b1])

    # Right edge (i=w-1)
    for j in range(h - 1):
        t = j * w + (w - 1)
        t1 = (j + 1) * w + (w - 1)
        b = n_top + j * w + (w - 1)
        b1 = n_top + (j + 1) * w + (w - 1)
        faces.append([t, t1, b])
        faces.append([t1, b1, b])

    faces = np.array(faces)

    print(f"Building mesh with {len(faces)} triangles...")
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = all_verts[f[j], :]

    stl_mesh.rotate([1, 0, 0], math.radians(180))

    stl_mesh.save(filepath)
    print(f"Saved: {filepath}")


### Main pipeline

print(f"Using device: {device}")

# 1. Load image
print("Loading image...")
img = cv2.imread(imgName)
if img is None:
    raise FileNotFoundError(f"Could not load image: {imgName}")
pil_image = Image.open(imgName).convert("RGB")

scale_ratio = math.sqrt(targetPixelCount / (img.shape[0] * img.shape[1]))

# Save step 1 outputs
cv2.imwrite(os.path.join(folder, "Step1_Original.jpeg"), img)
resized = resize(img, scale_ratio)
cv2.imwrite(os.path.join(folder, "Step1_Resized.jpeg"), resized)

pil_resized = resize_pil(pil_image, scale_ratio)
target_w, target_h = pil_resized.size

show_and_confirm("Step 1: Loaded & Resized", [
    ("Original", img),
    ("Resized", resized),
])

# 2. Bilateral filter preprocessing
print("Applying bilateral filter...")
denoised_bgr = cv2.bilateralFilter(resized, bilateralD, bilateralSigmaColor, bilateralSigmaSpace)
cv2.imwrite(os.path.join(folder, "Step2_Preprocessed.jpeg"), denoised_bgr)

denoised_pil = Image.fromarray(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))

show_and_confirm("Step 2: Bilateral Filter", [
    ("Before", resized),
    ("After", denoised_bgr),
])

# 3. Depth Anything V2 only
da_depth = run_depth_anything_v2(denoised_pil)
da_depth_resized = cv2.resize(da_depth, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
da_norm = normalize(da_depth_resized)
save_depth_as_image(da_norm, os.path.join(folder, "Step3_DepthAnythingV2.jpeg"))

show_and_confirm("Step 3: Depth Estimation", [
    ("Depth Anything V2", da_norm),
])

# 4. Canny edge detection and composite with depth
print("Running Canny edge detection...")
canny_edges = run_canny_edges(denoised_bgr, cannyThreshold1, cannyThreshold2, cannyGaussianBlur)
canny_resized = cv2.resize(canny_edges, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
save_depth_as_image(canny_resized, os.path.join(folder, "Step4_Canny.jpeg"))

composite = merge_depth_and_canny(da_norm, canny_resized, cannyBlendWeight)
save_depth_as_image(composite, os.path.join(folder, "Step4_Composite.jpeg"))

show_and_confirm("Step 4: Canny + Depth Composite", [
    ("Depth", da_norm),
    ("Canny", canny_resized),
    ("Composite", composite),
])

# 5. TV denoising after composite (smooth printable surface)
print("Applying TV denoising...")
smoothed = denoise_tv_chambolle(composite, weight=tvWeight)
smoothed = normalize(smoothed)
save_depth_as_image(smoothed, os.path.join(folder, "Step5_Depth_Smoothed.jpeg"))
save_thermal(smoothed, os.path.join(folder, "Step5_Depth_Thermal.jpeg"))

show_and_confirm("Step 5: TV Denoising", [
    ("Before (composite)", composite),
    ("After (smoothed)", smoothed),
])

# 6. Export to STL (simplified mesh via subsampling)
if not createComposite:
    stl_path = os.path.join(folder, "Step6_Model.stl")
    save_heightmap_stl(smoothed, stl_path, pixelSizeMM, maxDepthMM, baseThicknessMM, subsample=stlSubsample, max_print_size_mm=maxPrintSizeMM)
else:
    mid_y, mid_x = smoothed.shape[0] // 2, smoothed.shape[1] // 2
    quadrants = {
        "TopLeft":     smoothed[:mid_y, :mid_x],
        "TopRight":    smoothed[:mid_y, mid_x:],
        "BottomLeft":  smoothed[mid_y:, :mid_x],
        "BottomRight": smoothed[mid_y:, mid_x:],
    }
    for name, quad in quadrants.items():
        quad_img_path = os.path.join(folder, f"Step6_{name}.jpeg")
        save_depth_as_image(quad, quad_img_path)
        stl_path = os.path.join(folder, f"Step6_Model_{name}.stl")
        save_heightmap_stl(quad, stl_path, pixelSizeMM, maxDepthMM, baseThicknessMM, subsample=stlSubsample, max_print_size_mm=maxPrintSizeMM)

print("Done.")
