from nltk.translate import bleu_score


def get_bleu_scores(hypothesis, refrences):
    return {"Bleu_1": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1.0]),
            "Bleu_2": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 2, 1. / 2]),
            "Bleu_3": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 3, 1. / 3, 1. / 3]),
            "Bleu_4": bleu_score.corpus_bleu(refrences, hypothesis, weights=[1. / 4, 1. / 4, 1. / 4, 1. / 4])
            }
