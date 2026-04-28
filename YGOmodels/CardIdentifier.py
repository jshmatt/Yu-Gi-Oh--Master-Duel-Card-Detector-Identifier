from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import faiss
import pandas as pd
from transformers import AutoImageProcessor, AutoModel
import cv2 as cv
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path

dir_path = str(Path(__file__).resolve().parent)

class LoRALayer(nn.Module):
  def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
    super().__init__()

    self.original_layer = original_layer  # layer with frozen weights

    in_dim = original_layer.in_features
    out_dim = original_layer.out_features

    # scaling factor
    self.alpha = alpha
    self.r = r

    # weights
    self.w_a = nn.Linear(in_dim, r, bias=False)
    self.w_b = nn.Linear(r, out_dim, bias=False)

    # init w_a with kaiming distribution
    # init w_b with zeros
    # LoRA starts at identity with this initial parameters
    nn.init.kaiming_uniform_(self.w_a.weight, a=np.sqrt(5))
    nn.init.zeros_(self.w_b.weight)

  def forward(self, x):
    return self.original_layer(x) + self.w_b(self.w_a(x)) * self.alpha/self.r
  
class Dinov2(nn.Module):
  def __init__(self, r: int = 4, alpha: float = 1.0):
    super(Dinov2, self).__init__()

    dinov2 = AutoModel.from_pretrained(dir_path + '/dinov2s')

    # freeze layers
    for param in dinov2.parameters():
      param.requires_grad = False

    # replace query and value layers of attention blocks with lora-modified layers
    for layer in dinov2.encoder.layer:
      attention = layer.attention.attention
      attention.query = LoRALayer(attention.query, r=r, alpha=alpha)
      attention.value = LoRALayer(attention.value, r=r, alpha=alpha)

    self.backbone = dinov2

  def forward(self, img):
    output = self.backbone(img, interpolate_pos_encoding=True)
    return output.last_hidden_state
  
class Dinov2withLORA(nn.Module):
  def __init__(self, r: int = 4, alpha: float = 1.0):
    super(Dinov2withLORA, self).__init__()
    
    self.backbone = Dinov2()
    
    dict_path = dir_path + '/weights/dino.pth'
    finetuned_dict = torch.load(dict_path)

    # saved weights include the weight of the AdaFace loss function that will not be used in feature extraction
    # saved weights keys are have additional prefix "backbone." because of AdaFace init
    filtered_dict = {}
    for item in finetuned_dict.items():
      k, v = item
      if k != 'weight':
        k = k.removeprefix('backbone.')
        filtered_dict[k] = v
    
    self.backbone.load_state_dict(filtered_dict)
    
  def forward(self, x):
    return self.backbone(x)
  
class IdentifyCard:
  """Identify a card from the artwork.
  """
  def __init__(self,
               faiss_path : str = '/card_data', # location of the faiss features
               device : str = 'cuda'
               ):
    
    self.device = device
    
    self.model = Dinov2withLORA().eval().to(device)
        
    self.show_imgs_with_names = None # to store figure with arts and their names
    self.show_train_with_names = None # show train images for comparison
    
    # Load preprocess
    self.preprocess = AutoImageProcessor.from_pretrained(dir_path + '/dinov2s')
    self.preprocess.size = {"height": 98, "width": 98}
    self.preprocess.crop_size = {"height": 98, "width": 98}
    
    fusion_faiss = faiss.read_index(dir_path + faiss_path + '/fusion-IP.faiss')
    link_faiss = faiss.read_index(dir_path + faiss_path + '/link-IP.faiss')
    monster_faiss = faiss.read_index(dir_path + faiss_path + '/monster-IP.faiss')
    spell_faiss = faiss.read_index(dir_path + faiss_path + '/spell-IP.faiss')
    synchro_faiss = faiss.read_index(dir_path + faiss_path + '/synchro-IP.faiss')
    trap_faiss = faiss.read_index(dir_path + faiss_path + '/trap-IP.faiss')
    xyz_faiss = faiss.read_index(dir_path + faiss_path + '/xyz-IP.faiss')
    
    fusion_list = pd.read_csv(dir_path + faiss_path + '/fusion-IP.csv')
    link_list = pd.read_csv(dir_path + faiss_path + '/link-IP.csv')
    monster_list = pd.read_csv(dir_path + faiss_path + '/monster-IP.csv')
    spell_list = pd.read_csv(dir_path + faiss_path + '/spell-IP.csv')
    synchro_list = pd.read_csv(dir_path + faiss_path + '/synchro-IP.csv')
    trap_list = pd.read_csv(dir_path + faiss_path + '/trap-IP.csv')
    xyz_list = pd.read_csv(dir_path + faiss_path + '/xyz-IP.csv')
    
    self.type_dict = {
      'fusion': (fusion_faiss, fusion_list),
      'link': (link_faiss, link_list),
      'monster': (monster_faiss, monster_list),
      'spell': (spell_faiss, spell_list),
      'synchro': (synchro_faiss, synchro_list),
      'trap': (trap_faiss, trap_list),
      'xyz': (xyz_faiss, xyz_list)
    }
    
    
    self.match_plot = None
    
  def get_search_vector(self, art_array : np.ndarray):
    img = Image.fromarray(art_array)
    img = self.preprocess(img, return_tensors='pt')['pixel_values'].to(self.device)
    
    with torch.no_grad():
      embeddings = self.model(img)
      embeddings = embeddings[:,0,:]

    embeddings = embeddings.detach().cpu().numpy()
    faiss.normalize_L2(embeddings) # normalize because we are using cosine similarity
    return np.array([embeddings.squeeze()])
  
  def single_card_search(self, search_vector, type):
    if type in self.type_dict.keys():
      (index, cards_df) = self.type_dict[type]
      D, I = index.search(search_vector, k=50)
      cards_df = cards_df.iloc[I[0]]
      cards_df = cards_df.reset_index(drop=True)
      card_names = cards_df[['index', 'cardName']]
      card_names['dinoScore'] = D[0]
    else:
      print('Invalid type')
      card_names = pd.DataFrame()
    return card_names
    
  def rerank_with_clip(self, query_array, type, card_names):
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_model.to(self.device)
    
    # query embeddings
    query_image = Image.fromarray(query_array)
    query_tensor = clip_preprocess(query_image).unsqueeze(0).to(self.device)
    with torch.no_grad():
        query_clip = clip_model.encode_image(query_tensor)
        query_clip = F.normalize(query_clip, dim=-1)
    
    card_images = []
    for card_name in card_names['cardName']:
      img = Image.open(dir_path + '/card_data/{}-cards/{}'.format(type, card_name))
      card_images.append(img)
    
    # top 50 embeddings
    scores = []
    for card_img in card_images:
        card_tensor = clip_preprocess(card_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            card_clip = clip_model.encode_image(card_tensor)
            card_clip = F.normalize(card_clip, dim=-1)
        scores.append((query_clip * card_clip).sum().item())
    
    card_names['clipScore'] = scores
    
    return card_names

  def single_card_identify(self, artworks_types_list : list):
    """Main identifier method.

    Args:
        artworks_types_list (list[array, str]): list containing the array representation of the artwork and the card type (monster, spell, link etc.)

    Returns:
        dict["canidates": list, "plot_b64": str, "plot": fig]: "candidates" contain the list of the names of the top five candidates, "plot_b64" is string representation of the figure of the results, and "plot" is the image representation of the figure. 
    """
    art_array, type = artworks_types_list
    
    art_array = cv.resize(art_array, (128,128), interpolation=cv.INTER_CUBIC) # imporves score
    
    search_vector = self.get_search_vector(art_array)
    card_names_dino = self.single_card_search(search_vector, type)
    card_names_clip = self.rerank_with_clip(art_array, type, card_names_dino)
    card_names_clip['totalScore'] = card_names_clip['dinoScore']*0.7 + card_names_clip['clipScore']*0.3
    card_names_clip = card_names_clip.sort_values(by='totalScore', ascending=False)

    results_list = []
    
    for i in range(0,5):
      new_dict = {"name":card_names_clip['cardName'].iloc[i].removesuffix('.jpg'),
                  "type":type,
                  "score":round(card_names_clip['totalScore'].iloc[i], 2)}
      results_list.append(new_dict)

    fig, axes = plt.subplots(1, 6, figsize=(15, 3.2))
    fig.patch.set_facecolor('#0c1117')
    
    # query
    axes[0].imshow(art_array)
    axes[0].set_title('Query', color='#e2e8f0', fontsize=11, pad=8, fontweight='bold')
    axes[0].axis('off')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#38bdf8')
        spine.set_linewidth(2)
    
    # results
    bar_colors = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']
    for i, card in enumerate(results_list):
      ax = axes[i+1]
      img = Image.open(dir_path + '/card_data/{}-cards/{}.jpg'.format(type, card['name']))
      img = img.resize((128,128))
      img_arr = np.asarray(img, dtype=np.uint8)
      ax.imshow(img_arr)
      ax.set_title(
          f"{card['name']}\n{card['score']:.3f}",
          color=bar_colors[i], fontsize=9, pad=6
      )
      ax.axis('off')
  
    plt.tight_layout(pad=1.4)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode()
  
    return {'candidates':results_list, 'plot_b64':plot_b64, 'plot':fig}