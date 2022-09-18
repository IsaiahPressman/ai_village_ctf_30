from PIL import Image as Im
import base64
from pathlib import Path
import numpy as np
import requests
import json


IMAGE_DIMS = (224, 224)

"""
img = Image.open("/kaggle/input/ai-village-ctf/hotterdog/chester.png")
img = img.resize(IMAGE_DIMS)
img_bytes = img.tobytes()
b64_img_bytes = base64.urlsafe_b64encode(img_bytes)

data = {
    "input": b64_img_bytes.decode()
}

score = requests.post("https://hotterdog.fly.dev/score", data=json.dumps(data))
print(score.content)
"""


def get_weighted_mean_img(img1, weight, img2):
    # assert 0 <= weight <= 1
    return (img1 * weight + img2 * (1 - weight)).astype(int)


def main() -> bool:
    IMAGE_DIMS = (224, 224)
    all_hotdog_paths = [
        p
        for p in Path(r"C:\Users\isaia\Documents\GitHub\competitions\ai_village_ctf\ai-village-ctf\hotterdog").glob("*.*")
        if "hotdog" in p.stem.lower()
    ]
    print(f"Using {len(all_hotdog_paths)} images: {[p.stem for p in all_hotdog_paths]}")
    all_hotdog_imgs = []
    for path in all_hotdog_paths:
        img = np.asarray(Im.open(path).resize(IMAGE_DIMS).convert('RGB'))
        # img = img * np.random.uniform(low=0.1, high=0.9, size=(224, 224, 3)).clip(0, 1)
        all_hotdog_imgs.append(img)

    for img_path in Path(r"C:\Users\isaia\Documents\GitHub\competitions\ai_village_ctf\chester_adversaries").glob("9_*.png"):
        img = Im.open(img_path)
        img_idxs = np.random.choice(len(all_hotdog_imgs), size=2, replace=False)
        hotdog_img_subset = [
            img
            for i, img in enumerate(all_hotdog_imgs)
            if i in img_idxs
        ]
        mean_hotdog_image = np.stack(hotdog_img_subset, axis=-1).mean(-1)
        img = get_weighted_mean_img(
            np.asarray(img),
            np.random.uniform(low=0.77, high=0.88),
            # np.random.uniform(low=0.75, high=1., size=(224, 224, 3)),
            mean_hotdog_image * np.random.lognormal(mean=-1, sigma=1., size=(224, 224, 3)).clip(0, 1),
            # mean_hotdog_image * np.random.uniform(low=0., high=1., size=(224, 224, 3)).clip(0, 1),
        ).astype(np.uint8)
        img = Im.fromarray(img)
        img = img.resize(IMAGE_DIMS).convert('RGB')
        img_bytes = img.tobytes()
        b64_img_bytes = base64.urlsafe_b64encode(img_bytes)
        try:
            r = requests.post("https://hotterdog.fly.dev/score", data=json.dumps({"input": b64_img_bytes.decode()}))
            print(img_path, r.text)
            if len(r.text) >= 128:
                img.save(r"C:\Users\isaia\Documents\GitHub\competitions\ai_village_ctf\ai-village-ctf\hotterdog\successful_adversary.png")
                return True
        except requests.exceptions.ConnectionError:
            print("Connection problems. Contact the CTF organizers.")

    return False


if __name__ == '__main__':
    while True:
        if main():
            break

    print("Done!")
