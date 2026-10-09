from torchvision.transforms import v2

class Letterbox(v2.Transform):
    def __init__(self, size=(256, 256), fill=0):
        super().__init__()

        self.target_h = size[0]
        self.target_w = size[1]
        self.fill = fill

    def _transform(self, inpt, params):
        return inpt

    def forward(self, image):
        # image is expected to be a TVTensor Image
        # target["boxes"] is a BoundingBoxes TVTensor

        _, h, w = image.shape

        target_h = self.target_h
        target_w = self.target_w

        # ----------------------------------------
        # 1. Calculate scale
        # ----------------------------------------

        scale = min(
            target_w / w,
            target_h / h,
        )

        new_w = round(w * scale)
        new_h = round(h * scale)

        # ----------------------------------------
        # 2. Resize image + boxes together
        # ----------------------------------------

        resize = v2.Resize(
            size=(new_h, new_w)
        )

        image = resize(image)

        # ----------------------------------------
        # 3. Calculate padding
        # ----------------------------------------
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        pad_right = target_w - new_w - pad_left
        pad_bottom = target_h - new_h - pad_top

        # torchvision Pad order:
        # [left, top, right, bottom]

        pad = v2.Pad(
            padding=[
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ],
            fill=self.fill,
        )

        image = pad(image)

        return image