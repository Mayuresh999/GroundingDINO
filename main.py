from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
import cv2

CONFIG_PATH = r"groundingdino\config\GroundingDINO_SwinT_OGC.py"
CHECK_POINT_PATH = r"weights\groundingdino_swint_ogc.pth"

model = load_model(CONFIG_PATH, CHECK_POINT_PATH)

# print(model)

IMAGE_PATH = r".asset\AM_03_09ee4601c24a5ae4.jpg"
TEXT_PROMPT = "person without microphone"
BOX_THRESHOL = 0.35
TEXT_THRESHOLD = 0.25

source, image = load_image(IMAGE_PATH)
boxes, accuracy, obj_mane = predict(
    model = model, 
    image = image,
    caption = TEXT_PROMPT,
    box_threshold = BOX_THRESHOL,
    text_threshold = TEXT_THRESHOLD
    )

print(boxes, accuracy, obj_mane)

annotated_image = annotate(image_source = source, boxes = boxes, logits = accuracy, phrases=obj_mane)

sv.plot_image(annotated_image, (16,16))