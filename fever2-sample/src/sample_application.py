import json
import logging
import os
from logging.config import dictConfig
from typing import Optional, Dict, Any

from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.models import load_archive
from allennlp.predictors import Predictor

from fever.api.web_server import fever_web_api
from fever.evidence.retrieval_methods.retrieval_method import RetrievalMethod
from fever.reader import FEVERDocumentDatabase


def predict_single(predictor, retrieval_method, instance):
    evidence = retrieval_method.get_sentences_for_claim(instance["claim"])

    test_instance = predictor._json_to_instance({"claim": instance["claim"], "predicted_sentences": evidence})
    predicted = predictor.predict_instance(test_instance)

    max_id = predicted["label_logits"].index(max(predicted["label_logits"]))

    return {
        "claim": instance["claim"],
        "predicted_label": predictor._model.vocab.get_token_from_index(max_id, namespace="labels"),
        "predicted_evidence": evidence
    }


def my_sample_fever():
    logger = logging.getLogger()
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
        'allennlp': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
    })

    logger.info("My sample FEVER application")
    config = json.load(open(os.getenv("CONFIG_PATH", "configs/predict_local.json")))

    # Create document retrieval model
    logger.info("Load FEVER Document database from {0}".format(config["database"]))
    db = FEVERDocumentDatabase(config["database"])

    logger.info("Load DrQA Document retrieval index from {0}".format(config['index']))
    retrieval_method = RetrievalMethod.by_name(
        "fever.evidence.retrieval_methods.top_docs.TopNDocsTopNSents"
    )(db, config["index"], config["n_docs"], config["n_sents"])

    # Load the pre-trained predictor and model from the .tar.gz in the config file.
    # Override the database location for our model as this now comes from a read-only volume
    logger.info("Load Model from {0}".format(config['model']))
    archive = load_archive(config["model"],
                           cuda_device=config["cuda_device"],
                           overrides='{"dataset_reader":{"database":"' + config["database"] + '" }}')
    predictor = Predictor.from_archive(archive, predictor_name="fever")

    # The prediction function that is passed to the web server for FEVER2.0
    def baseline_predict(instances: Optional[Dict[str, Any]] = None, get_from_text=False, text=None):
        if get_from_text:
            if not text:
                raise ValueError("If 'get_from_text':bool is True, 'text':str must not be empty.")
            sentence_splitter = SpacySentenceSplitter(rule_based=True)
            instances = [{"id": idx, "claim": sentence} for idx, sentence in
                         enumerate(sentence_splitter.split_sentences(text))]

        # TODO: implement batched version
        predictions = []
        for instance in instances:
            predictions.append(predict_single(predictor, retrieval_method, instance))
        return predictions

    return fever_web_api(baseline_predict)
