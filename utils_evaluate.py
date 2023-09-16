import os
import json
import argparse
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry

warnings.filterwarnings('ignore')

def get_scores_coco(results_ans, results_rationale, results_reference, data_file):

    results_df = pd.DataFrame({
        "qid": list(results_ans.keys()),
        "prediction": list(results_rationale.values()),
        "reference": list(results_reference.values()),
    })

    results_data = dict(zip(results_df['qid'], results_df['prediction']))
    reference_data = dict(zip(results_df['qid'], results_df['reference']))
    # calculate metrics
    bleu1 = caculate_bleu(results_data, reference_data, gram=1)
    bleu4 = caculate_bleu(results_data, reference_data, gram=4)
    rouge = caculate_rouge(results_data, reference_data)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
    similarity = caculate_similariry(results_data, reference_data, model)

    scores = {
        "bleu1": bleu1 * 100,
        "bleu4": bleu4 * 100,
        "rouge": rouge * 100,
        "similarity": similarity * 100,
    }

    return scores