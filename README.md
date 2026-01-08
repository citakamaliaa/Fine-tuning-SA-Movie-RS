# Fine-Tuning Transformers for Sentiment Analysis for Addressing the Cold Start Problem in Movie Recommender Systems

## Project Description
This repository contains the implementation of a hybrid movie recommendation system designed to address cold-start problems and data sparsity. The system integrates sentiment analysis by fine-tuning Transformer models (BERT, RoBERTa, DistilBERT) on user reviews collected from X (Twitter).

The core innovation lies in converting unstructured text reviews into sentiment-based pseudo-ratings to enrich sparse user-item interaction data. The final recommendations are generated using a Weighted Hybrid Model that combines Collaborative Filtering (CF) and Content-Based Filtering (CBF).

## Key Features
- Sentiment-Based Pseudo-Ratings: Transforms unstructured social media reviews into implicit numerical ratings (scaled 0-5) to mitigate data sparsity.
- Transformer Fine-Tuning: Implements fine-tuned versions of BERT, RoBERTa, and DistilBERT for high-accuracy sentiment classification.
- Weighted Hybrid Architecture: Fuses Item-Based CF and Content-Based Filtering using an optimized weighting parameter ($\alpha$) to balance accuracy, diversity, and novelty.
- Cold-Start Robustness: Proven effectiveness in providing relevant recommendations for users with limited or no prior interaction history.

## Datasets
This study utilizes three primary datasets:
1. Movie Dataset: Metadata for 854 films (genres, descriptions, keywords) collected from IMDb.
2. Reviews Dataset: 34,086 tweets/reviews collected from X (Twitter) using Indonesian search queries translated to English.
3. Derived Rating Matrix: A generated user-item matrix where explicit ratings are replaced/augmented by sentiment scores derived from fine-tuned RoBERTa model.


## Methodology & Architecture
The system follows the pipeline illustrated below:
1. Text Preprocessing: Cleaning (emoticons, links), tokenization, and stemming.
2. Sentiment Classification: Fine-tuning RoBERTa to classify reviews and generate sentiment scores.
3. Rating Normalization: Applying Min-Max normalization to align sentiment scores with the recommendation rating scale (e.g., 1-5).
$$s_{ui}^{*}=\frac{s_{ui}-s_{min}}{s_{max}-s_{min}}\times(r_{max}-r_{min})+r_{min}$$
4. Hybrid Filtering: Calculating the final predicted rating using the weighted linear combination of CF and CBF:
$$\hat{r}_{ui}^{hybrid}=\alpha\cdot\hat{r}_{ui}^{CBF}+(1-\alpha)\cdot\hat{r}_{ui}^{CF}$$(Where $\alpha=0.1$ yielded the best performance).

## Experimental Results
1. Sentiment Analysis Performance
| Model | Accuracy |
| :--- | :--- |
| CNN-LSTM | 86.00% |
| Fine-tune BERT | 87.46% |
| Fine-tune RoBERTa | **90.11%** |
| Fine-tune DistilBERT | 89.33% |

2. Recommender System Performance
The Hybrid Filtering approach significantly outperformed standalone baselines, achieving the lowest error rates and the best balance of diversity and novelty:
| Model | MAE | RMSE | Diversity | Novelty |
| :-- | :-- | :-- | :-- | :-- |
| Item-based CF | 0.1949 |	0.2415	| 0.0210	| 10.1030|
| Content-based | 0.1164	| 0.1672	| 0.0114	| 10.9171 |
| Hybrid Filtering | **0.0452** | **0.0482** | **0.02131** | **0.1086** |

Impact: The hybrid model reduced MAE by 76.8% compared to Item-Based CF and 61.2% compared to Content-Based Filtering.

## Installation & Usage
1. Clone the repository
```git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
2. Install dependencies
```pip install -r requirements.txt```
3. Run the analysis
- Open the Sentiment Analysis notebook (Fine_tune_BERT.ipynb) and other related source file to train/evaluate the Transformer models.
- Open Recommender System notebook (Collaborative Filtering, Content-based Filtering, and Hybrid Filtering) ex: CF_BERT.ipynb or CBF.ipynb or Hybrid_Filtering.ipynb to run the hybrid filtering algorithm and generate recommendations.

## Citation
If you use this code or dataset in your research, please cite our paper:
@inproceedings{citakamalia2025finetuning,
  title={Fine-Tuning Transformers for Sentiment Analysis for Addressing the Cold Start Problem in Movie Recommender Systems},
  author={Citakamalia and Setiawan, Erwin Budi},
  booktitle={2025 3rd International Conference on Software Engineering and Information Technology (ICOSEIT)},
  year={2025},
  organization={IEEE}
}
