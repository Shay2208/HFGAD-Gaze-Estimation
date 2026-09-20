import cv2
import matplotlib.pyplot as plt
import numpy as np


def grad_cam_gaze_estimation(model, images, target_layer, gaze_direction=None):
    target_module = dict(model.named_modules())[target_layer]

    feature_map = []
    gradient = []

    def forward_hook(module, input, output):
        feature_map.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradient.append(grad_out[0])

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_backward_hook(backward_hook)

    images = images.requires_grad_()
    outputs = model(images)

    if gaze_direction is None:
        gaze_direction = outputs.squeeze(0).cpu().detach().numpy()

    target_score_pitch = outputs[:, 0]
    target_score_pitch.backward(retain_graph=True)
    feature_map_pitch = feature_map[0].squeeze(0).cpu().detach().numpy()
    gradient_pitch = gradient[0].squeeze(0).cpu().detach().numpy()

    weights_pitch = np.mean(gradient_pitch, axis=(1, 2))
    cam_pitch = np.zeros(feature_map_pitch.shape[1:], dtype=np.float32)
    for i, weight in enumerate(weights_pitch):
        cam_pitch += weight * feature_map_pitch[i]

    cam_pitch = np.maximum(cam_pitch, 0)
    cam_pitch = cv2.resize(cam_pitch, (images.shape[2], images.shape[3]))
    cam_pitch = (cam_pitch - cam_pitch.min()) / (cam_pitch.max() - cam_pitch.min() + 1e-8)

    target_score_yaw = outputs[:, 1]
    target_score_yaw.backward()
    feature_map_yaw = feature_map[0].squeeze(0).cpu().detach().numpy()
    gradient_yaw = gradient[0].squeeze(0).cpu().detach().numpy()

    weights_yaw = np.mean(gradient_yaw, axis=(1, 2))
    cam_yaw = np.zeros(feature_map_yaw.shape[1:], dtype=np.float32)
    for i, weight in enumerate(weights_yaw):
        cam_yaw += weight * feature_map_yaw[i]

    cam_yaw = np.maximum(cam_yaw, 0)
    cam_yaw = cv2.resize(cam_yaw, (images.shape[2], images.shape[3]))
    cam_yaw = (cam_yaw - cam_yaw.min()) / (cam_yaw.max() - cam_yaw.min() + 1e-8)

    combined_cam = (cam_pitch + cam_yaw) / 2
    combined_cam = np.maximum(combined_cam, 0)
    combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)

    heatmap_combined = cv2.applyColorMap(np.uint8(255 * combined_cam), cv2.COLORMAP_JET)
    ori_image = images.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    ori_image = (ori_image - ori_image.min()) / (ori_image.max() - ori_image.min())
    ori_image = np.uint8(255 * ori_image)
    superimposed_img_combined = cv2.addWeighted(
        cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR),
        0.7,
        heatmap_combined,
        0.3,
        0,
    )

    forward_handle.remove()
    backward_handle.remove()

    return combined_cam, superimposed_img_combined


def heatmap(spatial_logits, images):
    show_img = spatial_logits.mean(dim=1).squeeze(0).cpu().detach().numpy().astype(np.float32)

    if show_img.max() - show_img.min() > 0:
        show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
    else:
        show_img = np.zeros_like(show_img)
    show_img = cv2.resize(show_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    ori_image = images.astype(np.float32)
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    if ori_image.max() > 1:
        ori_image = ori_image / 255.0
    if ori_image.max() - ori_image.min() > 0:
        ori_image = (ori_image - ori_image.min()) / (ori_image.max() - ori_image.min())
    else:
        ori_image = np.zeros_like(ori_image)

    colorized = cv2.applyColorMap(np.uint8(255 * show_img), cv2.COLORMAP_JET)
    colorized = colorized[..., [2, 1, 0]]

    superimposed_img = cv2.addWeighted(
        cv2.cvtColor(np.uint8(255 * ori_image), cv2.COLOR_RGB2BGR),
        0.75,
        colorized,
        0.25,
        0,
    )
    return superimposed_img


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    if gaze.size == 2:
        gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
        gaze_gt[1] = -np.sin(gaze[1])
        gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    elif gaze.size == 3:
        gaze_gt = gaze

    return gaze_gt


def angular(gaze, label):
    assert gaze.size == 3, "The size of gaze must be 3"
    assert label.size == 3, "The size of label must be 3"

    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


def CropImg(img, X, Y, W, H):
    """
    X, Y is the coordinate of the left-top corner.
    W, H is width and height.
    """

    y_lim, x_lim = img.shape[0], img.shape[1]
    H = min(H, y_lim)
    W = min(W, x_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > x_lim:
        X = x_lim - W

    if Y + H > y_lim:
        Y = y_lim - H

    return img[Y : (Y + H), X : (X + W)]


def gazeVisual(img, gaze, label, start=None, color=(255, 0, 0), scale=250, thickness=3, tiplength=0.1):
    if len(gaze) == 2:
        gaze = gazeto3d(gaze)

    gaze_x = gaze[0] * scale
    gaze_y = gaze[1] * scale
    label_x = label[0] * scale
    label_y = label[1] * scale

    if start is None:
        start = (img.shape[1] // 2, img.shape[0] // 2)
    else:
        start = (int(start[0]), int(start[1]))

    gaze_end = (int(start[0] + gaze_x), int(start[1] + gaze_y))
    label_end = (int(start[0] + label_x), int(start[1] + label_y))

    cv2.arrowedLine(img, start, label_end, (0, 255, 0), thickness, 0, 0, tiplength)
    img_label = img.copy()
    cv2.arrowedLine(img, start, gaze_end, color, thickness, 0, 0, tiplength)

    return img, img_label
