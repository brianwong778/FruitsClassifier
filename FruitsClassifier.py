import torch
import torch.nn as nn
import torchvision.transforms as tt
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = pymongo.MongoClient(MONGODB_URI)
db = client[DB_NAME]
fruits_collection = db[COLLECTION_NAME]

# Define the model class
class FruitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 

            # Second Convolution Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 

            # Third Convolution Block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), 

            # Adaptive Pooling
            nn.AdaptiveAvgPool2d((1, 1)),

            # Flattening 
            nn.Flatten(),

            # Fully Connected Layer
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),    
            nn.Linear(256, 131) 
        )
    
    def forward(self, xb):
        return self.network(xb)

# Load the model
model = FruitsModel()
model.load_state_dict(torch.load('fruits_model.pth', map_location=torch.device('cpu')))
model.eval()


def preprocess_image(uploaded_file):
    transform = tt.Compose([
        tt.Resize(100),          # Resize the smaller side to 100 pixels
        tt.ToTensor(),
    ])
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    return transform(image)


def predict_image(img, model, class_names):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return class_names[preds[0].item()]

class_names = [
    'Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 
    'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 
    'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 
    'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 
    'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 
    'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 
    'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 
    'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 
    'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 
    'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 
    'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 
    'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 
    'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 
    'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 
    'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 
    'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 
    'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 
    'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 
    'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 
    'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 
    'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 
    'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 
    'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 
    'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 
    'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 
    'Tomato not Ripened', 'Walnut', 'Watermelon'
]


st.title('Fruit Classifier by Brian Wong')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)

    img = preprocess_image(uploaded_file)
    predicted_class = predict_image(img, model, class_names)
    st.write(f'Predicted Fruit: {predicted_class}')

    # Fetch fruit description from MongoDB
    fruit_info = fruits_collection.find_one({"name": predicted_class})
    if fruit_info and "description" in fruit_info:
        st.write(fruit_info["description"])
    else:
        st.write("Description not found.")

