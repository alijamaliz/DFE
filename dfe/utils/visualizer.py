import cv2
import numpy as np


def save_visualized_colmap_output(
        img1_path,
        img2_path,
        pts,
        batch_idx,
        output_dir = "/content/visualizations",
    ):
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

    _, img1_width, _ = img1.shape

    final_image = np.concatenate((img1, img2), axis=1)
    color = (0, 0, 255)
    for point in pts:
        x1 = int(point[0]) * 2
        y1 = int(point[1]) * 2
        x2 = (int(point[2]) * 2) + img1_width
        y2 = int(point[3]) * 2
        
        # Image 1 points
        final_image = cv2.circle(final_image, (x1, y1), radius=4,
                                    color=color, thickness=-1)

        # Image 2 points
        final_image = cv2.circle(final_image, (x2, y2), radius=4,
                                    color=color, thickness=-1)

        # Connector line
        final_image = cv2.line(final_image, (x1, y1),
                                (x2, y2), color, thickness=2)

    final_image_name = str(batch_idx + 1).zfill(5)
    cv2.imwrite(
        "{}/{}.jpg".format(output_dir, final_image_name), final_image)
