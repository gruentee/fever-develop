FEVER_ROOT=`pwd`
python -m fever.evidence.retrieve \
    --index $FEVER_ROOT/data/index/fever-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
    --in-file $FEVER_ROOT/data/fever-data/dev.jsonl \
    --out-file $FEVER_ROOT/data/fever/train.ns.pages.p1
