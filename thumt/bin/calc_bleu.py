from nltk.translate.bleu_score import sentence_bleu
import argparse

parser = argparse.ArgumentParser(
    description="Calculate the BLEU between the lines of two files.",
    usage="bert_tokenizerpython.py [<args>] [-h | --help]"
)

parser.add_argument("--translation", type=str,
                    help="Path of translation file")
parser.add_argument("--reference", type=str,
                    help="Path of reference file")
parser.add_argument("--output", type=str, default="bleu.out",
                    help="Path of output file")

args = parser.parse_args()

with open(args.translation) as f:
    translation = [line.strip().split(" ") for line in f]
    translation = sorted(translation)

with open(args.reference) as f:
    reference = [line.strip().split(" ") for line in f]
    reference = sorted(reference)

output = []
for trans, ref in zip(translation, reference):
    score = sentence_bleu([ref], trans)
    output.append((score, trans, ref))

output = sorted(output)
with open(args.output, "w") as f:
    for score, trans, ref in output:
        f.write("%f\n%s\n%s\n\n" % (score, " ".join(trans), " ".join(ref)))
