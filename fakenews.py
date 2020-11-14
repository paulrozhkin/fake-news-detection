import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from preprocess import preprocess_text


def load_specific_model():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained("./saved_model/")
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #config = BertConfig("./saved_model/config.json")
    #model = BertForSequenceClassification(config)

    model = torch.load("model_after_train.pt")

    #model = BertForSequenceClassification.from_pretrained("./saved_model/")
    #model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = model.to(device)
    #model.load_state_dict(torch.load("./saved_model/pytorch_model.bin"))
    #model.config.num_labels = 1

    #for param in model.parameters():
    #    param.requires_grad = False

    model.eval()

    return device, model, tokenizer

def detect_fake(text, device, model, tokenizer):
        text_parts = preprocess_text(text, device, tokenizer)
        overall_output = torch.zeros((1, 2)).to(device)
        try:
            for part in text_parts:
                if len(part) > 0:
                    overall_output += model(part.reshape(1, -1))[0]
        except RuntimeError:
            print("GPU out of memory, skipping this entry.")

        overall_output = F.softmax(overall_output[0], dim=-1)

        value, result = overall_output.max(0)

        term = "fake"
        if result.item() == 0:
            term = "real"

        print("{} at {}%".format(term, value.item() * 100))
        return term, value.item() * 100