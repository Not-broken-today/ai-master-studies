# my_model_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTranslationModel(nn.Module):
    def __init__(self, inp_voc, out_voc, emb_size=64, hid_size=256):
        super().__init__()
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        
        # 1. Bidirectional Encoder
        self.enc = nn.LSTM(emb_size, hid_size, bidirectional=True, batch_first=True)
        
        # 2. Bahdanau Attention
        self.attn_W_h = nn.Linear(hid_size, hid_size, bias=False)
        self.attn_W_s = nn.Linear(hid_size * 2, hid_size, bias=False)
        self.attn_v = nn.Linear(hid_size, 1, bias=False)
        
        # 3. Decoder LSTMCell
        self.dec_cell = nn.LSTMCell(emb_size + hid_size * 2, hid_size)
        
        # 4. Initial state projection
        self.dec_start_h = nn.Linear(hid_size * 2, hid_size)
        self.dec_start_c = nn.Linear(hid_size * 2, hid_size)
        
        self.logits = nn.Linear(hid_size, len(out_voc))

    def encode(self, inp, **flags):
        inp_emb = self.emb_inp(inp)
        enc_out, _ = self.enc(inp_emb)  # [batch, time, 2*hid]

        end_index = infer_length(inp, self.inp_voc.eos_ix, include_eos=False)
        end_index = torch.clamp(end_index, max=inp.shape[1] - 1)
        batch_idx = torch.arange(enc_out.shape[0], device=enc_out.device)
        enc_last = enc_out[batch_idx, end_index.detach(), :]

        dec_start_h = self.dec_start_h(enc_last)
        dec_start_c = self.dec_start_c(enc_last)
        return [dec_start_h, dec_start_c, enc_out]

    def _compute_attention(self, dec_h, enc_out):
        attn_h = self.attn_W_h(dec_h).unsqueeze(1)
        attn_s = self.attn_W_s(enc_out)
        score = torch.tanh(attn_h + attn_s)
        attn_weights = F.softmax(self.attn_v(score).squeeze(2), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
        return context

    def decode(self, prev_state, prev_tokens, **flags):
        prev_h, prev_c, enc_out = prev_state
        prev_emb = self.emb_out(prev_tokens)
        context = self._compute_attention(prev_h, enc_out)
        dec_input = torch.cat([prev_emb, context], dim=1)
        new_h, new_c = self.dec_cell(dec_input, (prev_h, prev_c))
        output_logits = self.logits(new_h)
        return [new_h, new_c, enc_out], output_logits

    def forward(self, inp, out, eps=1e-30, **flags):
        device = next(self.parameters()).device
        batch_size = inp.shape[0]
        bos = torch.tensor([self.out_voc.bos_ix] * batch_size, dtype=torch.long, device=device)
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        
        hid_state = self.encode(inp, **flags)
        for x_t in out.transpose(0, 1)[:-1]:
            hid_state, logits = self.decode(hid_state, x_t, **flags)
            logits_seq.append(logits)
            
        return F.log_softmax(torch.stack(logits_seq, dim=1), dim=-1)

    def translate(self, inp, greedy=False, max_len=None, eps=1e-30, **flags):
        device = next(self.parameters()).device
        batch_size = inp.shape[0]
        if max_len is None:
            max_len = 2 * inp.shape[1]
            
        bos = torch.tensor([self.out_voc.bos_ix] * batch_size, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, dtype=torch.uint8, device=device)
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        out_seq = [bos]
        
        hid_state = self.encode(inp, **flags)
        
        while True:
            hid_state, logits = self.decode(hid_state, out_seq[-1], **flags)
            if greedy:
                _, y_t = torch.max(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                y_t = torch.multinomial(probs, 1).squeeze(1)
                
            logits_seq.append(logits)
            out_seq.append(y_t)
            mask &= y_t != self.out_voc.eos_ix
            
            if not mask.any():
                break
            if len(out_seq) >= max_len:
                break
                
        return torch.stack(out_seq, dim=1), F.log_softmax(torch.stack(logits_seq, dim=1), dim=-1)


# ==========================================
# Utility functions (identical to basic_model_torch)
# ==========================================
def infer_mask(seq, eos_ix, batch_first=True, include_eos=True, dtype=torch.float):
    assert seq.dim() == 2
    is_eos = (seq == eos_ix).to(dtype=torch.float)
    if include_eos:
        if batch_first:
            is_eos = torch.cat((is_eos[:, :1] * 0, is_eos[:, :-1]), dim=1)
        else:
            is_eos = torch.cat((is_eos[:1, :] * 0, is_eos[:-1, :]), dim=0)
    count_eos = torch.cumsum(is_eos, dim=1 if batch_first else 0)
    mask = count_eos == 0
    return mask.to(dtype=dtype)

def infer_length(seq, eos_ix, batch_first=True, include_eos=True, dtype=torch.long):
    mask = infer_mask(seq, eos_ix, batch_first, include_eos, dtype)
    return torch.sum(mask, dim=1 if batch_first else 0)

def to_one_hot(y, n_dims=None):
    y_tensor = y.data
    y_tensor = y_tensor.to(dtype=torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot