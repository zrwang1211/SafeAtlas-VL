from ordinal_safety_vlm import SafetyPredictor

predictor = SafetyPredictor(
    "zrwang1211/SafeAtlas-Guard-2B",
    device_map="auto",
    dtype="bfloat16",
)
prediction = predictor.predict(
    image="examples/example.jpg",
    target_name="response",
    request="What fruit is shown in the image?",
    response="The image shows a red apple.",
)
print(prediction.to_dict())
