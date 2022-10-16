import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor:
    def __call__(self, image):
        return F.to_tensor(image)


class Resize:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise TypeError("'size'的类型应该是int，tuple或list其中之一")

    def __call__(self, image):
        return F.resize(image, size=self.size)