# Named Entity Recognition - Interview Projekt


## Task List
Task: Build and evaluate a simple NER model on a small dataset
- Use a pretrained transformer model from Hugging Face Transformers.
- Fine-tune the model on a small annotated dataset.
- Predict named entities on a few test sentences.
- Report precision, recall, and F1-score for the model.


## 📋 Project Summary
This project is about designing a clean, scalable ML workflow based on OOP and SOLID principles.
#### The gist:
- Named Entity Recognition: Identifying and categorizing textual entities (names, locations, etc.).
- Transformers: Interprets words based on their context (e.g., decides whether "Apple" is the fruit or the company).
- BIO schema: Standard tagging method for marking entity boundaries:
  - B (Beginning): The beginning of the entity.
  - I (Inside): The internal part of the entity.
  - O (Outside): Not an entity.

   
## Modell selection: distilbert-base-cased
First choice was BERT, because I wanted to maximize the accuracy.
Later I switched to DistilBERT instead of standard BERT, since the aim of the project fast effectiveness and response speed.
DistilBERT is smaller with 40%, faster with 60% and the accuracy is almost the same.
This means that the model delivers the expected results consistently with lower resource requirements, which is a critical factor in scalability.
Since the project is about entity recognition it was important to use "cased" model, to distinguish between upper and lower case letters. 

| Feature | BERT (base-cased) | DistilBERT (base-cased) | Difference / Advantage |
| :--- | :---: | :---: | :--- |
| **Number of parameters** | 110M | 66M | **~40% smaller** |
| **Inference speed** | 1.0x | 1.6x | **60% faster** |
| **Memory requirements (VRAM)** | ~4-6 GB | ~2-3 GB | **Half the RAM** is enough |
| **Training time** | Slow | Fast | More efficient iteration |
| **Accuracy (NER)** | 100% | ~97-98% | Negligible difference |


## Dataset selection: conll2003
This is the standard "benchmark" dataset for NER tasks. 
It contains the most important categories (Person, Organization, Location, Mixed). 
Despite its small size, it is excellent for fine-tuning.


## Challanges and solutions
- Hardware dependency: The code recognize the hardware, so the code can run on all platform.
- Training Time Optimization: Set up batch_size, because during the training got out of memor error. 
Since the entire NER dataset can contain tens of thousands of lines, the size for test train data is limited so training in faster and errors would occur faster.
- Human-Readable Label Mapping: Accelerated development using dataset subsampling for speed and label mapping to ensure human-readable model outputs.
- Out Of Memory Error: MPS memory errors on Apple Silicon by disabling the GPU allocation limit (HIGH_WATERMARK_RATIO=0.0), reducing the batch size, and implementing automated cache clearing.
  
## Training and Results
#### The results on the different size of datasets:

- 1500
```
Overall Precision : 0.6811
Overall Recall    : 0.6918
Overall F1        : 0.6864
Evaluation Loss   : 0.2292

Input: Apple was founded by Steve Jobs in California.
Output: 
[{'entity_group': 'ORG', 'score': 0.29259166, 'word': 'Apple', 'start': 0, 'end': 5}, 
{'entity_group': 'PER', 'score': 0.7899486, 'word': 'Steve Job', 'start': 21, 'end': 30}, 
{'entity_group': 'LOC', 'score': 0.9392351, 'word': 'California', 'start': 35, 'end': 45}]

Input: Budapest is the capital of Hungary.
Output: 
[{'entity_group': 'LOC', 'score': 0.92684305, 'word': 'Budapest', 'start': 0, 'end': 8}, 
{'entity_group': 'LOC', 'score': 0.9340516, 'word': 'Hungary', 'start': 27, 'end': 34}]

Input: Elon Musk bought Twitter and renamed it to X.
Output: 
[{'entity_group': 'ORG', 'score': 0.7634474, 'word': 'El', 'start': 0, 'end': 2}, 
{'entity_group': 'ORG', 'score': 0.60428685, 'word': 'Mu', 'start': 5, 'end': 7}, 
{'entity_group': 'ORG', 'score': 0.46659768, 'word': 'Twitter', 'start': 17, 'end': 24}, 
{'entity_group': 'ORG', 'score': 0.41412136, 'word': 'X', 'start': 43, 'end': 44}]
```

- Full dataset
```
Overall Precision : 0.9166
Overall Recall    : 0.9207
Overall F1        : 0.9186
Evaluation Loss   : 0.0630
Training Time: 

Input: Apple was founded by Steve Jobs in California.
Output: 
[{'entity_group': 'ORG', 'score': 0.9647108, 'word': 'Apple', 'start': 0, 'end': 5}, 
{'entity_group': 'PER', 'score': 0.9791188, 'word': 'Steve Jobs', 'start': 21, 'end': 31}, 
{'entity_group': 'LOC', 'score': 0.9992692, 'word': 'California', 'start': 35, 'end': 45}]

Input: Budapest is the capital of Hungary.
Output: 
[{'entity_group': 'LOC', 'score': 0.9989293, 'word': 'Budapest', 'start': 0, 'end': 8}, 
{'entity_group': 'LOC', 'score': 0.9989963, 'word': 'Hungary', 'start': 27, 'end': 34}]

Input: Elon Musk bought Twitter and renamed it to X.
Output: 
[{'entity_group': 'ORG', 'score': 0.98474985, 'word': 'Elon Musk', 'start': 0, 'end': 9}, 
{'entity_group': 'ORG', 'score': 0.9849696, 'word': 'Twitter', 'start': 17, 'end': 24}, 
{'entity_group': 'ORG', 'score': 0.972469, 'word': 'X', 'start': 43, 'end': 44}]
```

## Visualizaton: 
#### Loss - This graph shows the "development" of the model.
![loss_curve.png](plots/loss_curve.png)
It is vissible, that the training loss drops sharply at the beginning,
then stabilizes below 0.25 after 500 steps. This means that the model quickly learned the basic patterns.
There is a spike around step 3000, it can be a more difficult batch or training speed changed, 
but the model corrected itself immediately. 

#### Entity Performance - shows the accuracy of the model is in each category.
![entity_performance.png](plots/entity_performance.png)

- test_PER (0.96): Almost perfect. It recognizes personal names with confidence.
- test_LOC (0.94): Also excellent.
- test_MISC (0.83): This is the weakest link. 
This is understandable, as the "Miscellaneous" category is the most confusing (everything that is not a person, place, or organization).

#### Confusion Matrix - what it mixes with what.
![confusion_matrix.png](plots/confusion_matrix.png)
- The main axis (dark blue): Brutally strong. Most values are above 0.9.
- Error analysis: I-MISC row has a value of 0.15 in the "O" (Outside) column. 
This means that it considers 15% of the interior of MISC entities to be plain text.