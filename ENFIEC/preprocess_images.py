from PIL import Image


# 在保持图片的长宽比列的前提下，放大或者缩小图片
def resize_maintain_aspect(image, desired_size):
    # add padding to maintain the aspect ratio 处理图像让其保留原始纵横比
    old_size = image.size
    ratio = float(desired_size) / max(old_size)  #
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)  # Image.ANTIALIAS：高质量插值
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im

