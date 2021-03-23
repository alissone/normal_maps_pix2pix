import numpy as np


def predict_single_image(image, model, preprocess):
    if isinstance(image, (np.ndarray, np.generic)):
        image = Image.fromarray(np.uint8(image))

    if preprocess:
        image = preprocess(image)

    # Reshape to B, C, W, H to feed into network
    if torch.cuda.is_available():
        image = image.cuda()
        image = model(image.view(1, *image.shape)).detach().cpu().numpy()
    else:
        image = model(image.view(1, *image.shape)).detach().numpy()

    # reshape back into original W, H, C image shape
    image = image.reshape(*image.shape[1:]).transpose(1, 2, 0)

    return image
