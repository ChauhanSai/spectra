from utils.saliency import get_salient_cells

img_path = "data/coco/train2017/000000000064.jpg"
most, least = get_salient_cells(img_path)

print("Most salient:", most)
print("Least salient:", least)
