import cv2

def resize_image(img, desired_size=224):
    old_size = img.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = [int(x*ratio) for x in old_size]

    if new_size[0] == 0:
        new_size[0] = 1

    if new_size[1] == 0:
        new_size[1] = 1

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = 0
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    padding = (top, bottom, left, right)
    return new_im, padding