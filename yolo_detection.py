import torch
import torch.serialization

# PATCH torch.load before model loading
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Safe globals for YOLO unpickling
try:
    import sys
    sys.path.append('./yolov5')
    from models.yolo import Model
    torch.serialization.add_safe_globals([Model])
except Exception:
    print("Warning: Could not import YOLO Model for safe globals - patch skipped")

import cv2
import os
from PIL import Image
from io import BytesIO
import re

one = [
    "", "one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight ",
    "nine ", "ten ", "eleven ", "twelve ", "thirteen ", "fourteen ", "fifteen ",
    "sixteen ", "seventeen ", "eighteen ", "nineteen "
]
ten = [
    "", "", "twenty ", "thirty ", "forty ", "fifty ", "sixty ", "seventy ",
    "eighty ", "ninety "
]

class CurrencyNotesDetection:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device:", self.device)

    def load_model(self, model_name):
        model = torch.hub.load('./yolov5', 'custom', path=model_name, source='local')
        return model

    def class_to_label(self, x):
        return self.classes[int(x)]

    def numToWords(self, n, s):
        result = ""
        if n > 19:
            result += ten[n // 10] + one[n % 10]
        else:
            result += one[n]
        if n != 0:
            result += s
        return result

    def convertToWords(self, n):
        out = ""
        out += self.numToWords((n // 10000000), "crore ")
        out += self.numToWords(((n // 100000) % 100), "lakh ")
        out += self.numToWords(((n // 1000) % 100), "thousand ")
        out += self.numToWords(((n // 100) % 10), "hundred ")
        if n > 100 and n % 100:
            out += "and "
        out += self.numToWords((n % 100), "")
        return out

    def get_text(self, labelCnt):
        if not labelCnt:
            return "No currency notes detected in the image"
        text = "Image contains"
        noOfLabels, counter = len(labelCnt), 0
        total_value = 0
        for k, v in labelCnt.items():
            match = re.search(r'\d+', k)
            if match:
                rupee_value = int(match.group())
            else:
                rupee_value = 0
            total_value += rupee_value * v
            if counter > 0:
                if counter == noOfLabels - 1:
                    text += " and"
                else:
                    text += ","
            text += f" {self.convertToWords(v)}{k.replace('Rupees', ' Rupee')}"
            if v > 1:
                text += "s"
            counter += 1
        text += f". Total value is {self.convertToWords(total_value)}Rupees ({total_value} Rs)."
        return text

    def get_detected_image(self, img):
        imgs = [img]
        results = self.model(imgs, size=416)
        results.print()
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        labelCnt = {}
        confidence_scores = {}
        for i in range(n):
            classLabel = self.classes[int(labels[i])]
            row = cord[i]
            confidence = float(row[4])
            print(f"{classLabel} is detected with {confidence:.2f} probability.")
            if classLabel in labelCnt:
                labelCnt[classLabel] += 1
                confidence_scores[classLabel].append(confidence)
            else:
                labelCnt[classLabel] = 1
                confidence_scores[classLabel] = [confidence]
        avg_confidence = {
            note_type: sum(scores) / len(scores)
            for note_type, scores in confidence_scores.items()
        }
        text = self.get_text(labelCnt)
        print(f"{text} This is from yolo_detection.py")
        results.render()
        return results.imgs[0], text, labelCnt, avg_confidence

def run_model(img):
    obj = CurrencyNotesDetection(
        model_name='./yolov5/runs/train/exp/weights/best.pt'
    )
    detected_img, detected_labels_text, raw_labels, confidence_scores = obj.get_detected_image(img)
    return detected_img, detected_labels_text, raw_labels, confidence_scores
