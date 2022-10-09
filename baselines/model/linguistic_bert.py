# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torchvision

def build_resnet(model_name, pretrained=True):
    cnn = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    return model

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

class LinguisticBERT(nn.Module):
    def __init__(self, BertTokenizer_CKPT="bert-base-uncased", BertModel_CKPT="bert-base-uncased", output_dim=100, max_len=25) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(BertTokenizer_CKPT)
        self.model = BertModel.from_pretrained(BertModel_CKPT)
        self.classifier = nn.Linear(768, output_dim)
        self.output_dim = output_dim
        self.max_len = max_len

    def forward(self, sentences):
        '''
        params: sentences: list of sentences in strings
        '''
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, max_length=self.max_len, truncation=True)
        # visual_embeds = torch.rand(1, 20, 2048)
        inputs.to(device)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        logits = self.classifier(last_hidden_states[:, 0, :])
        return logits # # B x output_dim


if __name__ == '__main__':
    model = LinguisticBERT(BertTokenizer_CKPT="/home/leiting/scratch/transformers_pretrained_models/",
                    BertModel_CKPT="/home/leiting/scratch/transformers_pretrained_models/",
                    output_dim=100).to(device)
    sentences = ['people\'s', 'hi, an apple?', 'where is the red flag?']
    logits = model(sentences)
    print(logits.shape)

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer.save_pretrained("/home/leiting/scratch/transformers_pretrained_models/")

    # model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    # model.save_pretrained("/home/leiting/scratch/transformers_pretrained_models/")