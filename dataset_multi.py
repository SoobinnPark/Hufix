import json
import torch
from PIL import Image
import torchvision.transforms.functional as F

###############################
class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=800, width=544, tokenizer=None): #
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.img_names = self.img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        sample = self.data[img_id]

        input_path = sample["image"]
        output_path = sample["target_image"]
        caption = sample["prompt"]

        ref_paths = sample.get("ref_images", [])


        if isinstance(ref_paths, str):
            ref_paths = [ref_paths]

        try:
            input_img = Image.open(input_path)
            output_img = Image.open(output_path)
            if ref_paths is not None:
                ref_imgs = [Image.open(ref_paths[i]) for i in range(len(ref_paths))]
            else:
                ref_imgs = None
           
        except Exception as e:
            print(f"[Dataset Warning] Failed to load image idx={idx} ({img_id}): {e}")
            return self.__getitem__((idx + 1) % len(self))

        def preprocess(img):
            t = F.to_tensor(img)
            t = F.resize(t, self.image_size)
            t = F.normalize(t, mean=[0.5]*3, std=[0.5]*3)
            return t

        input_t = preprocess(input_img)
        output_t = preprocess(output_img)
        # print("init: ", input_t.shape)
        output_list = [output_t]
        input_list = [input_t]

        if ref_imgs is not None:
            for ref_img in ref_imgs:
                ref_t_i = preprocess(ref_img)
                input_list.append(ref_t_i) # stack every ref_image
                output_list.append(ref_t_i)
                
            input_t = torch.stack(input_list, dim=0)
            output_t = torch.stack(output_list, dim=0)
            # print(input_t.shape)

        else:
            input_t = input_t.unsqueeze(0)
            output_t = output_t.unsqueeze(0)
        
        # print(input_t.shape)
       
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
            ).input_ids[0]
            out["input_ids"] = input_ids

        return out
