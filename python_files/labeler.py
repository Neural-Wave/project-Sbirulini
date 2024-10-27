from __future__ import annotations

import os
from PIL import ImageTk
from tkinter import Tk, Button, Label, Toplevel, Frame
import json
from typing import Union, List
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2


class ImageTransformer:
    def __init__(self):
        pass

    def transform(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError()


class IdentityTransformer(ImageTransformer):
    def transform(self, image: Image.Image) -> Image.Image:
        return image


class UnsharpMask(ImageTransformer):
    def transform(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        return image.filter(ImageFilter.UnsharpMask).crop((width // 2, 0, width, height))


class EdgeDetectionTransformer(ImageTransformer):
    def transform(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        return image.convert("L").filter(ImageFilter.UnsharpMask).crop((width // 2, 0, width, height))


class Denoising(ImageTransformer):

    def transform(self, image: Image.Image) -> Image.Image:
        gray_img = pil_to_open_cv(image)
        width, height = image.size
        denoised_img = cv2.fastNlMeansDenoising(gray_img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return Image.fromarray(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)).crop((width // 2, 0, width, height))


def pil_to_open_cv(pil_image):
    open_cv_image = np.array(pil_image.convert('RGB'))
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def canny_edge(img):
    img = pil_to_open_cv(img)
    t_lower = 250  # Lower Threshold 250 for good light
    t_upper = 300  # Upper threshold 300 for good light
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying the Canny Edge filter
    edge = cv2.Canny(img, t_lower, t_upper)
    color_coverted = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)


class CannyEdgeDetector(ImageTransformer):

    def transform(self, image: Image.Image) -> Image.Image:
        gray_img = pil_to_open_cv(image)
        width, height = image.size
        denoised_img = cv2.fastNlMeansDenoising(gray_img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        image =  Image.fromarray(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
        # width, height = image.size
        return canny_edge(image).crop((width // 2, 0, width, height))


labels = [
    'background',
    'only bars',
    'only stopper',
    'aligned',
    'not_aligned',
    'skip', # not sure
    'remove' # used to remove blurred and bad quality images
]


class Labeler:
    def __init__(self, target_size: Union[tuple[int, int], None] = (500, 500)):
        # Initialize the Tk root only once
        self.root = Tk()
        self.root.withdraw()  # Hide the root window
        self.selected_label = None
        self.target_size = target_size  # Desired image size, e.g., (200, 200)

    def display(self, images: List[Image.Image]) -> str:
        self.top = Toplevel(self.root)
        wf1 = self.top.winfo_screenwidth()
        hf1 = self.top.winfo_screenheight()
        A = str(wf1)
        B = str(hf1)
        self.top.geometry(A + "x" + B)
        self.selected_label = None

        if len(images) == 0:
            raise ValueError("Expected at least one image for display")

        frames = [Frame(self.top) for _ in images]
        for i, frame in enumerate(frames):
            frame.pack(side="left", padx=10, pady=10)

            img = images[i]
            if self.target_size:
                width, height = img.size
                if i == 0:
                    width, height = img.size
                    img = img.resize((int(width / 1.2), int(height / 1.2)))
                else:
                    img = img.resize((width // 2, height // 2))
            img_tk = ImageTk.PhotoImage(img)


            img_label = Label(frame, image=img_tk)
            img_label.image = img_tk
            img_label.pack()

        def set_label(label: str):
            self.selected_label = label
            self.top.destroy()

        for label in labels:
            btn = Button(self.top, text=label, command=lambda lbl=label: set_label(lbl))
            btn.pack(side="bottom", padx=5, pady=5)

        self.root.wait_window(self.top)

        return self.selected_label


def label(path, i):
    json_file = 'label_bad_light.json' if 'bad' in path else 'label_good_light.json'
    if os.path.exists(json_file):
        with open(json_file, 'r+') as f:
            content = f.read()
        json_content = json.loads(content)
    else:
        json_content = {}

    # function to create a better image
    transforms: list[ImageTransformer] = [
        Denoising(),
        CannyEdgeDetector()
        # EdgeDetectionTransformer(),
        # IdentityTransformer()
    ]
    assert isinstance(i, int)
    labeler = Labeler()
    files = sorted(os.listdir(path))
    count = 0
    for file in files:
        if file.endswith('jpg'):
            count += 1
            if count < i:
                continue
            full_path = f'{path}/{file}'
            # print(file.strip().lower() in json_content.keys())
            if file.strip().lower() in json_content.keys():
                pass
            else:
                print(full_path)
                image: Image.Image = Image.open(full_path)
                images_to_show: list[Image.Image] = []
                images_to_show.append(image)
                for t in transforms:
                    images_to_show.append(t.transform(image))
                label = labeler.display(images_to_show).lower().strip()
                assert label in labels
                if label == 'skip':
                    continue
                json_content[file] = label
                with open(json_file, 'w+') as f:
                    json.dump(json_content, f)


if __name__ == '__main__':
    path = './data/train_set/good_light'
    label(path, 9340)
