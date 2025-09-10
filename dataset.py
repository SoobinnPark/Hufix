import json
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=800, width=544, tokenizer=None): #
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

        self.img_names = self.img_ids #

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if idx >= len(self.img_ids):
            raise IndexError("Index out of range")

        img_id = self.img_ids[idx]
        sample = self.data[img_id]

        input_path = sample["image"]
        output_path = sample["target_image"]
        ref_path = sample.get("ref_image", None)
        caption = sample["prompt"]

        try:
            input_img = Image.open(input_path).convert("RGB")
            output_img = Image.open(output_path).convert("RGB")
            if ref_path is not None:
                ref_img = Image.open(ref_path).convert("RGB")
            else:
                ref_img = None
        except Exception as e:
            print(f"[Dataset Warning] Failed to load images for idx={idx} ({img_id}): {e}")
            return self.__getitem__((idx + 1) % len(self.img_ids))  # 순환 호출

        # transforms
        input_t = F.to_tensor(input_img)
        input_t = F.resize(input_t, self.image_size)
        input_t = F.normalize(input_t, mean=[0.5]*3, std=[0.5]*3)

        output_t = F.to_tensor(output_img)
        output_t = F.resize(output_t, self.image_size)
        output_t = F.normalize(output_t, mean=[0.5]*3, std=[0.5]*3)

        if ref_img is not None:
            ref_t = F.to_tensor(ref_img)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5]*3, std=[0.5]*3)

            input_t = torch.stack([input_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)
        else:
            input_t = input_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)

        out = {
            "conditioning_pixel_values": input_t,
            "output_pixel_values": output_t,
            "caption": caption,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out
