from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Crop(px=(0, 12)),  # crop images from each side by 0 to 12px (randomly chosen)

    ],
    random_order=True,
)
