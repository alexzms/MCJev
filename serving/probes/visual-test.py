"""End-to-end image test against the running server: synthetic shapes (known truth) + a photo URL."""
import os, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # serving/
from client import post, show, image_source
import client


def synthetic_shapes(size=256):
    """A red triangle (left) and a blue circle (right) on white, as a PNG data URI (known ground truth)."""
    import base64, io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    d.polygon([(size * 0.10, size * 0.75), (size * 0.30, size * 0.25), (size * 0.50, size * 0.75)], fill="red")
    d.ellipse([size * 0.58, size * 0.35, size * 0.90, size * 0.67], fill="blue")
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

url, auth = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
client.AUTH = auth
shapes = synthetic_shapes()
cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
req = {"states": [
  {"id": "shapes", "state": "Image 1 is a simple synthetic picture on a white background.", "images": [shapes],
   "questions": {
     "red_shape": {"type": "choice", "instructions": "Which shape in Image 1 is red?",
                   "criteria": {"triangle": "a triangle", "circle": "a circle", "square": "a square"}},
     "has_circle": {"type": "boolean", "instructions": "Does Image 1 contain a circle?"},
     "has_square": {"type": "boolean", "instructions": "Does Image 1 contain a square?"},
     "count": {"type": "choice", "instructions": "How many shapes are in Image 1?",
               "criteria": {"one": "exactly one shape", "two": "exactly two shapes", "three": "exactly three shapes"}},
     "circle_side": {"type": "choice", "instructions": "On which side of Image 1 is the blue shape?",
                     "criteria": {"left": "left half", "right": "right half"}}}},
  {"id": "cat", "state": "Image 1 is a photo.", "images": [cat],
   "questions": {
     "animal": {"type": "choice", "instructions": "Which animal is in Image 1?",
                "criteria": {"cat": "a cat", "dog": "a dog", "bird": "a bird", "none": "no animal"}},
     "outdoors": {"type": "boolean", "instructions": "Was Image 1 taken outdoors?"},
     "cuteness": {"type": "score", "instructions": "How cute is the subject of Image 1?",
                  "criteria": ["not cute", "somewhat cute", "cute", "very cute"]}}},
  {"id": "two_images", "state": "Two images are attached.", "images": [shapes, cat],
   "questions": {
     "which_photo": {"type": "choice", "instructions": "Which image is a photograph?",
                     "criteria": {"first": "Image 1", "second": "Image 2"}},
     "same": {"type": "boolean", "instructions": "Do Image 1 and Image 2 show the same thing?"}}},
  {"id": "text_only", "state": "x = 3", "questions": {"pos": {"type": "boolean", "instructions": "Is x positive?"}}},
]}
body, dt = post(url, req); show(body); print(f"-- round trip {dt*1000:.0f} ms")
