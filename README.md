# Art Similarity

## Summary  
  I have developed an image similarity search system for paintings using deep learning. It extracts feature embeddings from each artwork with a pretrained ResNet-50 model. This README describes the dataset, and usage of the project in detail and also the evaluation metrics, and visual examples of query results. The goal is a clear, reproducible reference for Art community.


## Motivation  
  Visual similarity search in art can help curate collections,to find related works, or assist art historians. Unlike hand-crafted descriptors, a deep model (ResNet50) can learn high-level visual features (style, composition, color) automatically. By Nature each person have intrest in Art , it is very exciting for me to work on this data and with this community.

## Key Features  
- **Deep CNN Encoder**: Uses a pretrained ResNet-50 from TorchVision【24†L100-L104】 to extract image features. 
- **Batch Processing & GPU**: Can batch-process images on GPU for fast embedding extraction【45†L353-L358】.  
- **Simple API**: Scripts and notebooks to extract features, build/search index, and visualize results with just a few commands.
- **Clean output**: The code output is simple and clean easily understandable.

## Model & Method Description  
We use a **pretrained ResNet-50** convolutional network (ImageNet weights) as the feature extractor【24†L100-L104】. Each painting is processed by the standard ResNet transforms: resize/crop, convert to tensor, and **ImageNet normalization** (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). In practice:


## Results & Visualization  
<img src="https://c8.alamy.com/comp/2PJYGTR/the-starry-night-vincent-van-gogh-2PJYGTR.jpg" height="300"> *Starry Night* (1889) by Van Gogh【14†L134-L136】 – The model finds paintings with similar swirling skies and color (e.g. other Van Gogh or Post-Impressionist works). It correctly identifies the style, retrieving paintings with comparable blue-yellow contrast.  

<img src="https://media.craiyon.com/2025-07-22/LVsj7FAVTz-3IX_ow4qaIg.webp" height="300"> *Mona Lisa* (1503–1506) by Leonardo da Vinci【18†L127-L129】 – As a famous Renaissance portrait, the model retrieves other 16th-century portraits and paintings with similar muted palette and composition. The example shows how it captures facial orientation and color tones to find matches.  

<img src="https://tse4.mm.bing.net/th/id/OIP.MfaNZWamZCjHXhC7qFsOlAHaKm?rs=1&pid=ImgDetMain&o=7&rm=3" height="300"> *Girl with a Pearl Earring* (ca.1665) by Vermeer【21†L127-L130】 – A Baroque-era portrait. The model retrieves other Dutch Golden Age portraits with similar lighting and composition. This demonstrates the encoder’s ability to capture fine details like background darkness and the subject’s gaze.

In each of the above examples, the cosine similarity (or L2 distance) between embeddings is used to rank images.


## Future Extensions  
Some possible improvements:  
- **Fine-tuning / Contrastive Learning:** Currently I use simple ImageNet features. Fine-tuning the encoder on art (or using contrastive methods like SimCLR) may improve art-specific similarity.  
- **Vision Transformers:** In future I will try a ViT (e.g. DINO or CLIP) for possibly better representations of paintings.  
- **Metadata Fusion:** Combining image features with artist or style metadata in a hybrid model or filter can give better accuracy.   
- **Deployment:** Package as a REST API or use TensorRT for fast embedding.  
- **Multi-modal:** Extending it to text+image search.  
