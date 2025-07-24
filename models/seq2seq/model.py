import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from .chat_dataset import ChatDataset
from .attention import LuongAttention
from .custom_types import Method, Token
from .vocab import Vocab
from .searchers import GreedySearch
import os
import random
from tqdm import tqdm


class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, embedding: nn.Embedding):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = embedding
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True) # batch_first is True, because I don't approve self-harm

    def forward(self, x, lengths):
        x = self.embedding(x) # Output shape: (batch_size, max_len_in_batch, hidden_size)
        packed_embedded = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:], hidden


class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, attn, embedding: nn.Embedding, dropout: int = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.attn = attn
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, last_hidden, encoder_outputs):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        decoder_outputs, hidden = self.rnn(embedded, last_hidden)
        attn_weights = self.attn(decoder_outputs, encoder_outputs)

        context = attn_weights.bmm(encoder_outputs).squeeze(1)

        concat_input = torch.cat((decoder_outputs.squeeze(1), context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)

        output = F.softmax(output, dim=1)
        return output, hidden


class Seq2SeqChatbot(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, encoder_num_layers: int, decoder_num_layers: int, decoder_embedding_dropout: float, device: torch.device):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.decoder_embedding_dropout = decoder_embedding_dropout
        self.vocab_size = vocab_size
        self.epoch = 0

        self.device = device
        self.vocab = Vocab([])
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn = LuongAttention(Method.DOT, hidden_size)
        self.encoder = Seq2SeqEncoder(hidden_size, hidden_size, encoder_num_layers, self.embedding)
        self.decoder = Seq2SeqDecoder(hidden_size, hidden_size, vocab_size, decoder_num_layers, self.attn, self.embedding, decoder_embedding_dropout)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.searcher = GreedySearch(self.encoder, self.decoder, self.embedding, device)
        self.to(device)
        self.eval_mode()

    def train(self, epochs, train_data, teacher_forcing_ratio, device, save_dir, model_name, clip, save_every):
        def maskNLLLoss(inp, target, mask):
            crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
            loss = crossEntropy.masked_select(mask).mean()
            loss = loss.to(device)
            return loss
        
        epoch_progress = tqdm(range(self.epoch, self.epoch + epochs), desc="Training", unit="epoch", leave=True)
        epoch_progress.set_description(f"maskNLLLoss: None")
        
        for epoch in epoch_progress:
            for x_train, y_train, x_lengths, y_mask in train_data:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                # Squeeze because batches are made in dataset and DataLoader is only for shuffling
                x_train = x_train.squeeze(0).to(device)
                y_train = y_train.squeeze(0).to(device)
                x_lengths = x_lengths.squeeze(0) # Lengths are computed on CPU
                y_mask = y_mask.squeeze(0).to(device)
        
                encoder_outputs, hidden = self.encoder(x_train, x_lengths) # Output shape: (batch_size, max_len_in_batch, hidden_size)
                hidden = hidden[:self.decoder_num_layers]
                loss = 0
                decoder_input = torch.LongTensor([[Token.BOS_TOKEN.value] for _ in range(y_train.shape[0])])
                decoder_input = decoder_input.to(device)
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    for t in range(y_train.shape[1]): # Process words in all batches for timestep t
                        decoder_outputs, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                        decoder_input = y_train[:, t].unsqueeze(1)
                        mask_loss = maskNLLLoss(decoder_outputs, y_train[:, t], y_mask[:, t])
                        loss += mask_loss
                else:
                    for t in range(y_train.shape[1]):
                        decoder_outputs, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                        decoder_input = torch.argmax(decoder_outputs, dim=1).unsqueeze(1)
                        mask_loss = maskNLLLoss(decoder_outputs, y_train[:, t], y_mask[:, t])
                        loss += mask_loss
        
                loss.backward()
        
                _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
                _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
        
            if (epoch % save_every == 0 and epoch != 0) or epoch == save_every - 1:
                directory = os.path.join(save_dir, model_name, '{}-{}'.format(self.encoder_num_layers, self.decoder_num_layers, self.hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'epoch': epoch + self.epoch,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.vocab.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))
            
            epoch_progress.set_description(f"maskNLLLoss: {loss:.8f}")
    
    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.embedding = self.embedding.to(device)
        self.attn = self.attn.to(device)

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.embedding.train()
        self.attn.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        self.embedding.eval()
        self.attn.eval()

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        encoder_sd = checkpoint["en"]
        decoder_sd = checkpoint["de"]
        embedding_sd = checkpoint["embedding"]
        self.vocab.__dict__ = checkpoint["voc_dict"]
        encoder_optimizer_sd = checkpoint["en_opt"]
        decoder_optimizer_sd = checkpoint["de_opt"]
        self.epoch = checkpoint["epoch"]

        self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        self.embedding.load_state_dict(embedding_sd)
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)

    def forward(self, input_seq: str):
        input_seq = ChatDataset._ChatDataset__normalize(input_seq)
        input_seq = self.vocab.sentence_indices(input_seq + ["<eos>"]).unsqueeze(0).to(self.device)
        output, _ = self.searcher(input_seq, torch.tensor(input_seq.shape[1]).view(1), 10)
        output = [self.vocab.index2word[i.item()] for i in output]
        output = [word for word in output if word not in ("<bos>", "<eos>", "<pad>")]
        return " ".join(output)
    

if __name__ == "__main__": # Run as module
    from .chat_dataset import ChatDataset
    import torch.utils.data as data

    CHAT_HISTORY_PATH = "models/seq2seq/data/train/chat_history.json"
    batch_size = 20
    chat_dataset = ChatDataset(CHAT_HISTORY_PATH, max_message_count=10_000, batch_size=batch_size)
    train_data = data.DataLoader(chat_dataset, batch_size=1, shuffle=True)

    device = torch.device("cpu")
    chatbot = Seq2SeqChatbot(500, chat_dataset.vocab.size, 2, 2, 0.1, device)
    chatbot.load_checkpoint("models/seq2seq/checkpoint/150_checkpoint.tar")
    chatbot.train_mode()
    chatbot.train(3, train_data, 0.5, device, "./checkpoint/temp/", "frantics_fox", 50.0, 100)