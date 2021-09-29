import argparse
import json
from fever.scorer import fever_score

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)
parser.add_argument("--predicted_evidence",type=str)
args = parser.parse_args()

predicted_labels =[]
predicted_evidence = []
actual = []

with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        predicted_labels.append(json.loads(line)["predicted_label"])


with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        line = json.loads(line)
        if "predicted_sentences" in line:
            predicted_evidence.append(line["predicted_sentences"])
        elif "predicted_evidence" in line:
            predicted_evidence.append(line["predicted_evidence"])
        elif "evidence" in line:
            all_evidence = []
            for evidence_group in line["evidence"]:
                all_evidence.extend(evidence_group)

            predicted_evidence.append(list(set([(evidence[2],evidence[3]) for evidence in all_evidence])))


predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

fever, acc, pr, rec, f1 = fever_score(predictions, actual)
print("FEVER Score: {}\n".format(fever))
print("Label Accuracy: {}\n".format(acc))
print("Evidence Precision: {}\n".format(pr))
print("Evidence F!: {}\n".format(rec))
print("Evidence F1: {}\n".format(f1))


