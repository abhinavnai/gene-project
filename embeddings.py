import csv
import torch
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# PREFIX → INFIX CONVERSION
# ---------------------------
def prefix_to_infix(expr):

    stack = []
    tokens = expr.split(',')[::-1]

    operators = {"+", "-", "*", "/", "^"}

    for token in tokens:

        if token in operators:
            a = stack.pop()
            b = stack.pop()
            stack.append(f"( {a} {token} {b} )")

        else:
            stack.append(token)

    return stack[0]


# ---------------------------
# LOAD MATHBERT
# ---------------------------
print("Loading MathBERT...")

tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
model = AutoModel.from_pretrained("tbs17/MathBERT")

device = torch.device("cpu")
model = model.to(device)

print("Using device:", device)


# ---------------------------
# DATASET
# ---------------------------
input_file = "strings_infix_all_four.csv"

embeddings_list = []


# ---------------------------
# PROCESS CSV
# ---------------------------
with open(input_file, "r") as f:

    reader = csv.reader(f)

    # skip header if present
    next(reader)

    for i, row in enumerate(reader):

        # columns 1-4 contain prefix expressions
        infix = [prefix_to_infix(col) for col in row[1:5]]

        # weighted expression
        expr = f"t1 * {infix[0]} + t2 * {infix[1]} + t3 * {infix[2]} + t4 * {infix[3]}"

        # tokenize
        inputs = tokenizer(
            expr,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # model inference
        with torch.no_grad():
            outputs = model(**inputs)

        # mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1)

        embeddings_list.append(embedding.cpu())

        if i % 100 == 0:
            print("Processed rows:", i)


# ---------------------------
# SAVE EMBEDDINGS
# ---------------------------
print("Saving embeddings...")

embeddings_tensor = torch.cat(embeddings_list, dim=0)

torch.save(embeddings_tensor, "mathbert_embeddings.pt")

print("Saved file: mathbert_embeddings.pt")
print("Embedding shape:", embeddings_tensor.shape)
