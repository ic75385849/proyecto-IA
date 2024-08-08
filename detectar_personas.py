import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

# Cargar el modelo preentrenado de detección de objetos
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Transformación de las imágenes
transform = T.Compose([
    T.ToTensor()
])

# Ruta de la carpeta con las imágenes
image_folder = 'img'

# Ruta de la carpeta para guardar las imágenes procesadas
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Umbral de confianza
confidence_threshold = 0.9

# Iterar sobre las imágenes en la carpeta
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Leer la imagen
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Añadir una dimensión para el batch

        # Realizar la detección
        with torch.no_grad():
            predictions = model(img_tensor)

        # Obtener las cajas de los rostros
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        labels = predictions[0]['labels']  # Etiquetas de las categorías

        # Dibujar las cajas sobre la imagen
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(boxes):
            if scores[i] >= confidence_threshold and labels[i] == 1:  # Umbral y filtro de categoría (1 es 'persona' en COCO)
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)

        # Guardar la imagen con las detecciones
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)
        print(f"Imagen guardada en: {output_path}")
