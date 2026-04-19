from ultralytics import YOLO
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pathlib import Path

dir_path = str(Path(__file__).resolve().parent)

class DetectCards:
  def __init__(self, 
               pickle_result : bool = False, # save results in pickle
               save_each_card : bool = False, # save image file for each detected card
               save_path : str = None): # indicate save path if pickle_result or save_each_card is True
    
    self.model = YOLO(dir_path + '/weights/yolo.pt')
    self.pickle_result = pickle_result
    self.save_each_card = save_each_card
    self.save_path = save_path
    
    self.boxes_plot = None # to store figure of image with bounding boxes
    self.cards_plot = None
    
    self.number_detected_cards = 0
    self.extracted_cards_list = []

  def detect_cards(self, img):
    result = self.model.predict(source = img,
                                conf = 0.1,
                                end2end = False, # enable NMS
                                iou = 0.1,
                                agnostic_nms = True)
        
    return result
  
  def deck_with_boxes(self, result):
    # color dictionary in bgr
    card_colors = {0: (150, 50, 180), 
                      1: (245, 135, 60), 
                      2: (0, 160, 255),
                      3: (150, 210, 170),
                      4: (220, 220, 220),
                      5: (150, 50, 220),
                      6: (120, 120, 120)}
    
    result = result[0]
    img = result.orig_img.copy()

    boxes = result.boxes.xyxy.cpu().numpy()  # left, top, right, bottom
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls in zip(boxes, confidences, class_ids):
      color = card_colors[cls]        

      start_point = (int(box[0]), int(box[1]))
      end_point = (int(box[2]), int(box[3]))
      cv.rectangle(img, start_point, end_point, color, 2)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert to rgb for matplotlib
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    
    plt.close()
    
    return fig
    
  def extract_each_card(self, result, card_save_path=None):
    result = result[0]

    img = result.orig_img.copy()
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # dictionary of class id to their actual names
    class_names = self.model.names
    
    cards_array = []
    types_array = []
    
    card_df = pd.DataFrame(columns=['xyxy', 'confidence', 'class'])
    pickle_i = 1
    
    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_ids)):
      card = img[int(box[1]) : int(box[3]),
                 int(box[0]) : int(box[2])]
      
      cls = class_names[cls]
      cls = cls.removesuffix(' card')
      
      if self.save_each_card:
        card_pil = Image.fromarray(card)
        card_pil.save(card_save_path + '/{}-{}.jpg'.format(i, cls))
      
      # convert to rgb
      card = card[:,:,::-1]
      cards_array.append(card)
      types_array.append(cls)

      if self.pickle_result:
        new_df = pd.DataFrame({'xyxy':[box],
                               'confidence':[conf],
                               'card_type':[cls]})
        card_df = pd.concat([card_df, new_df])
        
    if self.pickle_result:
      if not os.path.exists(self.save_path + '/detected-cards.pkl'):
        card_df.to_pickle(self.save_path + '/detected-cards.pkl')
      
      else:
        while os.path.exists(self.save_path + '/detected-cards-{}.pkl'.format(pickle_i)):
          pickle_i += 1
        card_df.to_pickle(self.save_path + '/detected-cards-{}.pkl'.format(pickle_i))
            
    return cards_array, types_array
    
  def plot_cards(self, cards_array, types_array):
    cards_number = len(cards_array)
    
    fig = plt.figure(figsize=(10,10))
    
    r = 7
    c = int(np.ceil(cards_number / r))
    
    for i, (card, type) in enumerate(zip(cards_array, types_array)):
      fig.add_subplot(r,c,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.title(type, fontsize=11)
      plt.imshow(card)
    
    plt.tight_layout()
    plt.close()

    report_number_of_cards = f"""Detected cards: {cards_number}
    Fusion: {types_array.count('fusion')}
    Link: {types_array.count('link')}
    Monster: {types_array.count('monster')}
    Spell: {types_array.count('spell')}
    Synchro: {types_array.count('synchro')}
    Trap: {types_array.count('trap')}
    XYZ: {types_array.count('xyz')}
    """
    
    self.number_detected_cards = report_number_of_cards
    return fig
    
  def predict(self, img):
    result = self.detect_cards(img)

    self.boxes_plot = self.deck_with_boxes(result)
      
    cards_array, types_array = self.extract_each_card(result)
        
    self.cards_plot = self.plot_cards(cards_array, types_array)
            
    results_list = []
    for image, type in zip(cards_array, types_array):
      new_dict = {"image":image.astype(np.uint8), "type":type}
      results_list.append(new_dict)
      
    return results_list