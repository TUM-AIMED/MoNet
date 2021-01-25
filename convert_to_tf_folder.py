from pathlib import Path
from tqdm import tqdm
from os import symlink


class SegmentationData:
    def __init__(
        self,
        overall_folder,
    ):
        mask_folders = {
            f.name.lower().replace("mask", ""): f
            for f in Path(overall_folder).iterdir()
            if f.is_dir() and "mask" in f.name.lower()
        }
        image_folders = {
            f.name.lower().replace("image", ""): f
            for f in Path(overall_folder).iterdir()
            if f.is_dir() and "image" in f.name.lower()
        }
        self.segmentation_list = []
        for folder_id, image_folder in image_folders.items():
            if not folder_id in mask_folders:
                print(f"Warning: Corresponding masks for {folder_id} not found")
            mask_folder = mask_folders[folder_id]
            images = {
                f.name.lower(): f
                for f in image_folder.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            }
            masks = {
                f.name.lower(): f
                for f in mask_folder.iterdir()
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            }
            for key, image in images.items():
                if key not in masks:
                    print(f"\tMask for {image} not found")
                self.segmentation_list.append((image, masks[key]))

    def __getitem__(self, index):
        image, mask = self.segmentation_list[index]
        # image, mask = Image.open(image), Image.open(mask)
        # image, mask = self.transform(image), self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.segmentation_list)


if __name__ == "__main__":
    overall_folder = Path("../data/tf_format").resolve()
    train_imgs_path = overall_folder / "train/images/segmentation"
    train_masks_path = overall_folder / "train/masks/segmentation"
    val_imgs_path = overall_folder / "val/images/segmentation"
    val_masks_path = overall_folder / "val/masks/segmentation"

    train_imgs_path.mkdir(parents=True, exist_ok=True)
    train_masks_path.mkdir(parents=True, exist_ok=True)
    val_imgs_path.mkdir(parents=True, exist_ok=True)
    val_masks_path.mkdir(parents=True, exist_ok=True)

    train_data = SegmentationData("../data/Train_PNGs")
    val_data = SegmentationData("../data/Test_PNGs")

    for i in tqdm(
        range(len(train_data)), total=len(train_data), desc="training data", leave=False
    ):
        train_img, train_mask = train_data[i]
        file_name = f"{i}.png"
        symlink(Path(train_img).resolve(), (train_imgs_path / file_name).resolve())
        symlink(Path(train_mask).resolve(), (train_masks_path / file_name).resolve())

    for i in tqdm(
        range(len(val_data)), total=len(val_data), desc="test data", leave=False
    ):
        val_img, val_mask = val_data[i]
        file_name = f"{i}.png"
        symlink(Path(val_img).resolve(), (val_imgs_path / file_name).resolve())
        symlink(Path(val_mask).resolve(), (val_masks_path / file_name).resolve())
