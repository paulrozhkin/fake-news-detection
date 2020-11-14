import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification

from preprocess import preprocess_text


def training_and_save_model():
    real_data = pd.read_csv('data/True.csv')
    fake_data = pd.read_csv('data/Fake.csv')

    nb_articles = min(len(real_data), len(fake_data))
    real_data = real_data[:nb_articles]
    fake_data = fake_data[:nb_articles]

    real_data['is_fake'] = False
    fake_data['is_fake'] = True

    data = pd.concat([real_data, fake_data])

    # Shuffle the data
    data = shuffle(data).reset_index(drop=True)
    data.head()

    train_data, validate_data, test_data = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])

    train_data = train_data.reset_index(drop=True)
    validate_data = validate_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    del real_data
    del fake_data

    print("Size of training set: {}".format(len(train_data)))
    print("Size of validation set: {}".format(len(validate_data)))
    print("Size of testing set: {}".format(len(test_data)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.config.num_labels = 1

    for param in model.parameters():
        param.requires_grad = False

    # Add three new layers at the end of the network
    model.classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
        nn.Softmax(dim=1)
    )

    model = model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

    print_every = 100

    total_loss = 0
    all_losses = []

    CUDA_LAUNCH_BLOCKING = 1

    model.train()

    for idx, row in train_data.iterrows():
        text_parts = preprocess_text(str(row['text']), device, tokenizer)
        label = torch.tensor([row['is_fake']]).long().to(device)

        optimizer.zero_grad()

        overall_output = torch.zeros((1, 2)).float().to(device)
        for part in text_parts:
            if len(part) > 0:
                try:
                    input = part.reshape(-1)[:512].reshape(1, -1)
                    # print(input.shape)
                    overall_output += model(input, labels=label)[1].float().to(device)
                except Exception as e:
                    print(str(e))

        #     overall_output /= len(text_parts)
        overall_output = F.softmax(overall_output[0], dim=-1)

        if label == 0:
            label = torch.tensor([1.0, 0.0]).float().to(device)
        elif label == 1:
            label = torch.tensor([0.0, 1.0]).float().to(device)

        # print(overall_output, label)

        loss = criterion(overall_output, label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if idx % print_every == 0 and idx > 0:
            average_loss = total_loss / print_every
            print("{}/{}. Average loss: {}".format(idx, len(train_data), average_loss))
            all_losses.append(average_loss)
            total_loss = 0

    torch.save(model, "model_after_train.pt")

    model.save_pretrained('./saved_model/')
    torch.save(model.state_dict(), './saved_model/model.pt')
    tokenizer.save_pretrained('./saved_model/')

    plt.plot(all_losses)

    total = len(test_data)
    number_right = 0
    model.eval()
    with torch.no_grad():
        for idx, row in test_data.iterrows():
            text_parts = preprocess_text(str(row['text']), device, tokenizer)
            label = torch.tensor([row['is_fake']]).float().to(device)

            overall_output = torch.zeros((1, 2)).to(device)
            try:
                for part in text_parts:
                    if len(part) > 0:
                        overall_output += model(part.reshape(1, -1))[0]
            except RuntimeError:
                print("GPU out of memory, skipping this entry.")
                continue

            overall_output = F.softmax(overall_output[0], dim=-1)

            result = overall_output.max(0)[1].float().item()

            if result == label.item():
                number_right += 1

            if idx % print_every == 0 and idx > 0:
                print("{}/{}. Current accuracy: {}".format(idx, total, number_right / idx))

    print("Accuracy on test data: {}".format(number_right / total))


def test_news(text, device, model, tokenizer):
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
