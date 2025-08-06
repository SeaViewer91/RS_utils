"""
blur_separator_gui.py
=====================

This module implements a graphical utility for classifying images into
"Blur" and "Normal" categories based on the visual sharpness of each
photograph.  It builds on the layout used for the previous blur
detection GUI but replaces the simplistic filename‑based labelling
mechanism with an actual image analysis routine.  Filenames still
serve as optional hints for calibrating the threshold but do not
determine the final classification.

Key features:

1. **Automatic sharpness measurement** – Each image is converted to
   grayscale and its gradient variance is computed.  Sharp (well
   focused) images exhibit larger gradient variances, whereas blurry
   images have smaller values.

2. **Dynamic threshold estimation** – When the user presses the
   "Analysis" button, the program measures all images and then
   computes a recommended threshold.  Training examples (those whose
   filenames contain user‑defined blur and normal keywords) are used
   to estimate separate means for blurry and normal images; the
   threshold is set to the midpoint between these means.  If no
   training examples are available the threshold defaults to the
   median of the measured metrics.

3. **Interactive reanalysis** – A numeric entry in the parameters
   panel allows users to override the automatically estimated
   threshold.  After entering a value and pressing "Reanalysis",
   classifications are recomputed accordingly.

4. **Manual override** – For each image the user can review the
   classification and change it via radio buttons.  Final
   classifications (including manual edits) are used when moving
   files.

5. **Sorting to subfolders** – Clicking the "Finish" button creates
   `Blur` and `Normal` subdirectories in the selected directory and
   moves files based on the chosen labels.

To run the application:

    python blur_separator_gui.py

The program requires Pillow and NumPy to be installed.
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np


def compute_gradient_variance(image: Image.Image) -> float:
    """Compute the variance of the gradient magnitude of a grayscale image.

    This function converts the provided PIL image to grayscale,
    computes the horizontal and vertical gradients using NumPy, then
    calculates the variance of the gradient magnitude.  Higher
    variances correspond to sharper images.

    Args:
        image: A PIL Image instance.
    Returns:
        A float representing the gradient magnitude variance.
    """
    # Convert to grayscale and to a float array
    gray = image.convert('L')
    arr = np.array(gray, dtype=float)
    # Compute gradients along y (rows) and x (cols)
    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(grad_mag.var())


class MainWindow(tk.Tk):
    """Primary window for selecting the directory and launching work window."""

    def __init__(self):
        super().__init__()
        self.title("Blur Classification Tool")
        self.geometry("520x140")
        self.resizable(True, False)

        # Directory selection
        self.path_var = tk.StringVar()
        tk.Label(self, text="Select directory containing images:").pack(pady=5)
        select_frame = tk.Frame(self)
        select_frame.pack(fill="x", padx=10)
        entry = tk.Entry(select_frame, textvariable=self.path_var)
        entry.pack(side="left", expand=True, fill="x")
        btn = tk.Button(select_frame, text="Browse", command=self.select_directory)
        btn.pack(side="left", padx=5)

        # Work window reference
        self.work_window = None

    def select_directory(self):
        """Open a file dialog to select the image directory and launch work window."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.path_var.set(directory)
            # Destroy existing work window if present
            if self.work_window and tk.Toplevel.winfo_exists(self.work_window):
                self.work_window.destroy()
            self.work_window = WorkWindow(self, directory)


class WorkWindow(tk.Toplevel):
    """Work window that performs analysis and allows the user to sort images."""

    def __init__(self, master, directory):
        super().__init__(master)
        self.title("작업 화면")
        self.directory = directory
        self.images = []  # paths of images
        self.metrics = []  # computed gradient variances
        self.labels = []  # predicted labels: 1=Blur, 0=Normal
        self.user_labels = []  # user‑edited labels
        self.index = 0

        # Keywords for identifying training examples
        self.blur_keyword = tk.StringVar(value="blur")
        self.normal_keyword = tk.StringVar(value="normal")
        # Threshold value used for classification
        self.threshold_var = tk.StringVar(value="")

        # Load image file list
        self.load_images()

        # GUI layout
        self.create_widgets()
        # Initial status update
        self.update_status()
        if self.images:
            self.show_image(0)

        # Keyboard navigation
        self.bind("<Control-Right>", lambda e: self.next_image())
        self.bind("<Control-Left>", lambda e: self.previous_image())

    def load_images(self):
        """Populate the images list with supported file types from the directory."""
        supported = {".jpg", ".jpeg", ".png", ".bmp"}
        for fname in os.listdir(self.directory):
            if os.path.splitext(fname)[1].lower() in supported:
                self.images.append(os.path.join(self.directory, fname))
        self.images.sort()
        # Allocate placeholder arrays for metrics and labels
        self.metrics = [None] * len(self.images)
        self.labels = [0] * len(self.images)
        self.user_labels = [0] * len(self.images)

    def create_widgets(self):
        """Set up the frames and widgets in the work window."""
        # Top frame: analysis button and summary
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=5)
        analyze_btn = tk.Button(top_frame, text="Analysis", command=self.run_analysis)
        analyze_btn.pack(side="left")
        self.status_var = tk.StringVar()
        tk.Label(top_frame, textvariable=self.status_var).pack(side="left", padx=10)

        # Center frame: left image, right crop + radio buttons
        center_frame = tk.Frame(self)
        center_frame.pack(fill="both", expand=True, padx=10, pady=5)
        # Left: image display
        left_frame = tk.Frame(center_frame)
        left_frame.pack(side="left", expand=True, fill="both")
        self.img_label = tk.Label(left_frame, bg="#e0e0e0")
        self.img_label.pack(expand=True, fill="both")
        # Right: crop + radio
        right_frame = tk.Frame(center_frame)
        right_frame.pack(side="left", fill="y", padx=10)
        self.crop_label = tk.Label(right_frame, bg="#e0e0e0")
        self.crop_label.pack(pady=5)
        self.radio_var = tk.IntVar(value=1)
        blur_radio = tk.Radiobutton(right_frame, text="Blur Image", variable=self.radio_var, value=1, command=self.update_user_label)
        blur_radio.pack(anchor="w")
        normal_radio = tk.Radiobutton(right_frame, text="Normal Image", variable=self.radio_var, value=0, command=self.update_user_label)
        normal_radio.pack(anchor="w")

        # Bottom frame: navigation, index status, parameters, finish
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill="x", padx=10, pady=5)

        # Index status
        self.index_status_var = tk.StringVar()
        tk.Label(bottom_frame, textvariable=self.index_status_var).pack(anchor="w")

        # Navigation buttons
        nav_frame = tk.Frame(bottom_frame)
        nav_frame.pack(side="left", pady=5)
        tk.Button(nav_frame, text="Previous", command=self.previous_image).pack(side="left", padx=5)
        tk.Button(nav_frame, text="Next", command=self.next_image).pack(side="left")

        # Parameters frame: blur/normal keywords and threshold entry
        param_frame = tk.Frame(bottom_frame)
        param_frame.pack(side="left", padx=10)
        tk.Label(param_frame, text="Blur Key:").grid(row=0, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.blur_keyword, width=10).grid(row=0, column=1, padx=5)
        tk.Label(param_frame, text="Normal Key:").grid(row=0, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.normal_keyword, width=10).grid(row=0, column=3, padx=5)
        tk.Label(param_frame, text="Threshold:").grid(row=1, column=0, sticky="e")
        self.threshold_entry = tk.Entry(param_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=1, column=1, padx=5)
        tk.Button(param_frame, text="Reanalysis", command=self.reanalysis).grid(row=1, column=2, columnspan=2, padx=5)

        # Finish button
        tk.Button(bottom_frame, text="Finish", command=self.finish_sorting).pack(side="right")

    def run_analysis(self):
        """Perform the initial sharpness analysis and determine a threshold."""
        if not self.images:
            messagebox.showinfo("No images", "There are no images to analyze.")
            return
        # Compute metrics for all images
        for i, path in enumerate(self.images):
            try:
                img = Image.open(path)
            except Exception:
                self.metrics[i] = None
                continue
            # Downsample large images for speed: limit largest side to 600 pixels
            w, h = img.size
            max_side = max(w, h)
            if max_side > 600:
                scale = 600.0 / max_side
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            self.metrics[i] = compute_gradient_variance(img)
        # Determine threshold using training examples
        blur_keys = self.blur_keyword.get().strip().lower().split(',') if self.blur_keyword.get().strip() else []
        normal_keys = self.normal_keyword.get().strip().lower().split(',') if self.normal_keyword.get().strip() else []
        blur_vals = []
        normal_vals = []
        for path, metric in zip(self.images, self.metrics):
            if metric is None:
                continue
            name = os.path.basename(path).lower()
            if any(key and key in name for key in blur_keys):
                blur_vals.append(metric)
            elif any(key and key in name for key in normal_keys):
                normal_vals.append(metric)
        if blur_vals and normal_vals:
            threshold = (np.mean(blur_vals) + np.mean(normal_vals)) / 2.0
        elif blur_vals:
            # Only blur examples: threshold slightly above max blur value
            threshold = np.max(blur_vals) * 1.2
        elif normal_vals:
            # Only normal examples: threshold slightly below min normal value
            threshold = np.min(normal_vals) * 0.8
        else:
            # Fallback: median of all metrics
            valid_metrics = [m for m in self.metrics if m is not None]
            threshold = np.median(valid_metrics) if valid_metrics else 0.0
        # Store and display threshold (rounded)
        self.threshold_var.set(f"{threshold:.2f}")
        # Classify images
        self.apply_threshold(threshold)
        self.show_image(self.index)

    def apply_threshold(self, threshold: float):
        """Assign labels based on the provided threshold."""
        for i, metric in enumerate(self.metrics):
            if metric is None:
                # Default unknown metrics to blur to encourage review
                self.labels[i] = 1
            else:
                self.labels[i] = 1 if metric < threshold else 0
        # Start user_labels from predicted labels only on first run or after reanalysis
        self.user_labels = list(self.labels)
        self.update_status()

    def reanalysis(self):
        """Reanalyse using an optional user‑specified threshold, or recompute one."""
        self.update_user_label()
        # If user has provided a numeric threshold, use it
        entry_val = self.threshold_var.get().strip()
        try:
            if entry_val:
                threshold = float(entry_val)
            else:
                # Otherwise compute anew (similar to run_analysis but without recomputing metrics)
                blur_keys = self.blur_keyword.get().strip().lower().split(',') if self.blur_keyword.get().strip() else []
                normal_keys = self.normal_keyword.get().strip().lower().split(',') if self.normal_keyword.get().strip() else []
                blur_vals = []
                normal_vals = []
                for path, metric in zip(self.images, self.metrics):
                    if metric is None:
                        continue
                    name = os.path.basename(path).lower()
                    if any(key and key in name for key in blur_keys):
                        blur_vals.append(metric)
                    elif any(key and key in name for key in normal_keys):
                        normal_vals.append(metric)
                if blur_vals and normal_vals:
                    threshold = (np.mean(blur_vals) + np.mean(normal_vals)) / 2.0
                elif blur_vals:
                    threshold = np.max(blur_vals) * 1.2
                elif normal_vals:
                    threshold = np.min(normal_vals) * 0.8
                else:
                    valid = [m for m in self.metrics if m is not None]
                    threshold = np.median(valid) if valid else 0.0
                self.threshold_var.set(f"{threshold:.2f}")
        except ValueError:
            messagebox.showerror("Invalid input", "Threshold must be a number.")
            return
        # Apply threshold
        self.apply_threshold(threshold)
        # Refresh current image display
        self.show_image(self.index)

    def update_status(self):
        """Update the summary labels for processed/suspected counts and index."""
        total = len(self.images)
        suspected = sum(self.user_labels)
        self.status_var.set(f"Processed: {total}/{total}  Suspected: {suspected}")
        if total:
            self.index_status_var.set(f"Suspected: {suspected}  Index: {self.index + 1}/{total}")
        else:
            self.index_status_var.set("No images loaded")

    def show_image(self, idx: int):
        """Display the image at position `idx` and update UI elements."""
        if not self.images:
            return
        self.index = max(0, min(idx, len(self.images) - 1))
        path = self.images[self.index]
        # Open image
        try:
            img = Image.open(path)
        except Exception:
            messagebox.showerror("Error", f"Cannot open image: {path}")
            return
        # Display scaled version in left pane
        max_size = (500, 400)
        display_img = img.copy()
        display_img.thumbnail(max_size, Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display_img)
        self.img_label.configure(image=self.tk_img)
        self.img_label.image = self.tk_img
        # Create crop (central 300x300) for right pane
        w, h = img.size
        crop_size = 300
        left = max((w - crop_size) // 2, 0)
        top = max((h - crop_size) // 2, 0)
        right = min(left + crop_size, w)
        bottom = min(top + crop_size, h)
        crop = img.crop((left, top, right, bottom))
        # If smaller than desired, pad with white
        if crop.size != (crop_size, crop_size):
            bg = Image.new("RGB", (crop_size, crop_size), (255, 255, 255))
            cw, ch = crop.size
            bg.paste(crop, ((crop_size - cw) // 2, (crop_size - ch) // 2))
            crop = bg
        self.tk_crop = ImageTk.PhotoImage(crop)
        self.crop_label.configure(image=self.tk_crop)
        self.crop_label.image = self.tk_crop
        # Set radio selection to user label
        self.radio_var.set(self.user_labels[self.index])
        self.update_status()

    def update_user_label(self):
        """Record the current radio selection into user_labels."""
        if self.images:
            self.user_labels[self.index] = self.radio_var.get()
            self.update_status()

    def next_image(self):
        """Advance to the next image, saving current selection."""
        if self.images and self.index < len(self.images) - 1:
            self.update_user_label()
            self.show_image(self.index + 1)

    def previous_image(self):
        """Go back to the previous image, saving current selection."""
        if self.images and self.index > 0:
            self.update_user_label()
            self.show_image(self.index - 1)

    def finish_sorting(self):
        """Move images to Blur/Normal subdirectories based on user labels."""
        if not self.images:
            messagebox.showinfo("No images", "No images to sort.")
            return
        blur_dir = os.path.join(self.directory, "Blur")
        normal_dir = os.path.join(self.directory, "Normal")
        os.makedirs(blur_dir, exist_ok=True)
        os.makedirs(normal_dir, exist_ok=True)
        # Save current selection before moving
        self.update_user_label()
        for path, label in zip(self.images, self.user_labels):
            dest_dir = blur_dir if label == 1 else normal_dir
            dest_path = os.path.join(dest_dir, os.path.basename(path))
            try:
                if os.path.abspath(path) != os.path.abspath(dest_path):
                    shutil.move(path, dest_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to move {path}: {e}")
        messagebox.showinfo("Finished", "Images sorted into 'Blur' and 'Normal' folders.")
        self.destroy()


def main():
    root = MainWindow()
    root.mainloop()


if __name__ == '__main__':
    main()
