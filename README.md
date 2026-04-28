# Details
This is a web-based app that detects and identifies a Yu-Gi-Oh! Master Duel card from a screenshot of the game using two machine learning architectures: YOLO26 and DinoV2. It is mainly used on a screenshot of a game **video/livestream** with the absence of official extensions to help viewers.

# Preliminaries
## 1. Python
You need Python to run the app. See https://www.python.org/downloads/.
## 2. Clone the repository
Navigate to a directory of your choice and run
```bash
git clone https://github.com/jshmatt/Yu-Gi-Oh--Master-Duel-Card-Detector-Identifier.git
```
## 3. DinoV2 repo
The card identifier uses the DinoV2-small architecture. See https://github.com/facebookresearch/dinov2 for installation instructions, and put the DinoV2 local repository inside the YGOmodels folder.
## 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 4. Install PyTorch
The PyTorch version will depend on your system. Set up your preference at https://pytorch.org/get-started/locally/ and run the command shown.

## 6. Download Model and Card Data
|Data|Download Link|
|-|-|
|Card index and key|https://huggingface.co/datasets/jshmatt/DinoV2-YGO-card-embeddings|
|Detector weights|https://huggingface.co/jshmatt/YOLO-YGO-card-detector/blob/main/yolo.pt|
|Identifier weights|https://huggingface.co/jshmatt/DInoV2-YGO-card-identifier/blob/main/dino.pth|
|Card images|https://ygoprodeck.com/api-guide/|

Ensure that the data is inside the YGOmodels folder and nested as shown below.
```
YGOmodels/
├──card_data/
   ├──fusion-cards/    # Card images from the API
   ├──link-cards/
   ├──monster-cards/
   ├──spell-cards/
   ├──synchro-cards/
   ├──trap-cards/
   ├──xyz-cards/
   ├──fusion-IP.csv     # Downloaded card key
   ├──fusion-IP.faiss   # Downloaded card index
   .          
   .
   .
   ├──xyz-IP.csv
   ├──xyz-IP.faiss
├──dinov2s/             # Cloned DinoV2-small repository
├──weights/
   ├──dino.pth          # Downloaded identifier weights
   ├──yolo.pt           # Downloaded detector weights
```
# How to Use
1. Run the [main Python script](app.py). 
2. Go to [localhost](http://localhost/8080). \
   ![upload](https://drive.google.com/uc?export=view&id=1mvGP2dpKrbsO0AcpTYe4ffhybia57klz) 
3. Upload a screenshot of the deck editor then run the detector. The screenshot should only contain the area bounded by the red box as shown below.\
      ![deckbox](https://drive.google.com/uc?export=view&id=1OrUP7QSzLdL67VpKrl91iFxnulPX2vyX) 
4. Select a card to identify.
   ![select](https://drive.google.com/uc?export=view&id=14YftxRPCiWcN9ntKAyVjlr478Z02LQGh) 
5. Crop only the artwork of the card then run the identifier. Refer to the red box in the image below. \
   ![crop](https://drive.google.com/uc?export=view&id=1DQxp77MDqI9I8j6M3-nY9xeUyQDz5gAH) 
6. See results. \
   ![results](https://drive.google.com/uc?export=view&id=1xDh0AZZaPNVGPdbCWporp44Qf3wXMcfV)
