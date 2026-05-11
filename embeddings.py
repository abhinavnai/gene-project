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

        token = token.strip()

        if token in operators:

            a = stack.pop()
            b = stack.pop()

            stack.append(f"( {a} {token} {b} )")

        else:
            stack.append(token)

    return stack[0]


# ---------------------------
# CHECK GPU
# ---------------------------
print("Checking GPU...")

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)

    device = torch.device("cuda")

else:

    print("GPU not found. Using CPU.")
    device = torch.device("cpu")


# ---------------------------
# LOAD MATHBERT
# ---------------------------
print("Loading MathBERT...")

tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")

model = AutoModel.from_pretrained("tbs17/MathBERT")

model = model.to(device)

model.eval()

print("Using device:", device)


# ---------------------------
# DATASET
# ---------------------------
input_file = "all_runs_full.csv"

embeddings_list = []


# ---------------------------
# PROCESS CSV
# ---------------------------
with open(input_file, "r", encoding="utf-8") as f:

    reader = csv.reader(f)

    # skip header
    next(reader)

    for i, row in enumerate(reader):

        try:

            # columns 2-5 contain prefix expressions
            infix = [
                prefix_to_infix(col)
                for col in row[1:5]
            ]

            # weighted expression
            expr = (
                f"t1 * {infix[0]} + "
                f"t2 * {infix[1]} + "
                f"t3 * {infix[2]} + "
                f"t4 * {infix[3]}"
            )

            # tokenize
            inputs = tokenizer(
                expr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # move tensors to GPU
            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            # inference
            with torch.no_grad():

                outputs = model(**inputs)

            # mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)

            # move back to CPU before saving
            embeddings_list.append(
                embedding.cpu()
            )

            if i % 100 == 0:

                print(f"Processed rows: {i}")

                if torch.cuda.is_available():

                    memory = torch.cuda.memory_allocated() / 1024**2

                    print(f"GPU Memory Used: {memory:.2f} MB")

        except Exception as e:

            print(f"Error at row {i}: {e}")


# ---------------------------
# SAVE EMBEDDINGS
# ---------------------------
print("Saving embeddings...")

embeddings_tensor = torch.cat(
    embeddings_list,
    dim=0
)

torch.save(
    embeddings_tensor,
    "new_mathbert_embeddings.pt"
)

print("Saved file: new_mathbert_embeddings.pt")

print("Embedding shape:", embeddings_tensor.shape)