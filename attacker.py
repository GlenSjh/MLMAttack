import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from classifiers import LSTM
from textattack.shared import WordEmbedding
from utils.data_helper import stop_words
from utils.sampling import dirichlet_softmax
from utils.data_collator import MyDataCollatorForWholeWordMask


class MLMAttacker(nn.Module):
    def __init__(self, model_name, num_labels, target_label):
        super(MLMAttacker, self).__init__()

        if (model_name == 'lstm'):
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
            self.mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
            self.model = LSTM(self.tokenizer.vocab_size, num_labels=num_labels)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            if (not hasattr(self.tokenizer, 'vocab')):
                self.tokenizer.vocab = self.tokenizer.get_vocab()
            self.mlm = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        self.filter_model = copy.deepcopy(self.mlm)
        self.mask_dc = MyDataCollatorForWholeWordMask(self.tokenizer, mlm_probability=0.15, return_tensors='pt')
        self.target_label = target_label
        self.prediction = False

    def set_temp(self, temp):
        self.N_TEMP = temp

    def get_masked_sentences_topk(self, sentence_ids, attention_mask, logits):
        with torch.no_grad():
            scores = self.filter_model(input_ids=sentence_ids, attention_mask=attention_mask).logits
        mask = scores >= torch.topk(scores, 100)[0][..., -1, None]
        mask = mask * (F.embedding(sentence_ids, self.mask_table))

        mask.scatter_(-1, sentence_ids.unsqueeze(-1), True)
        logits = logits.masked_fill(~mask, -math.inf)

        return logits

    def get_mask_table(self):
        print("Building mask table...")

        counter_fitted = WordEmbedding.counterfitted_GLOVE_embedding()
        emb = torch.tensor(counter_fitted.embedding_matrix)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sim = torch.matmul(emb, emb.T)
        sim = sim > 0.1

        cf_to_tokenizer_vocab = {}
        tokenizer_to_cf_vocab = {}
        for word, idx in counter_fitted._word2index.items():
            if (word in self.tokenizer.vocab):
                tokenizer_idx = self.tokenizer.vocab[word]
                cf_to_tokenizer_vocab[idx] = tokenizer_idx
                tokenizer_to_cf_vocab[tokenizer_idx] = idx

        mask_table = torch.ones(self.tokenizer.vocab_size, self.tokenizer.vocab_size, dtype=torch.bool)

        word_list = []
        for src_word in self.tokenizer.vocab.keys():
            if (isinstance(self.tokenizer, transformers.AlbertTokenizer) or isinstance(self.tokenizer, transformers.AlbertTokenizerFast)):
                if (not src_word.startswith('▁')):  # subword for AlbertTokenizer
                    word_list.append('##' + src_word)
                else:
                    word_list.append(src_word)
            else:
                word_list.append(src_word)

        for idx, word in tqdm(enumerate(word_list)):
            if (idx in tokenizer_to_cf_vocab):
                cf_idx = tokenizer_to_cf_vocab[idx]
                repl_idx = sim[cf_idx].nonzero().squeeze()
                if (repl_idx.shape):
                    sim_repl = set(
                        cf_to_tokenizer_vocab[x.item()] for x in repl_idx if (x.item() in cf_to_tokenizer_vocab))
                else:
                    sim_repl = set([idx])

                mask_table[idx][list(set(tokenizer_to_cf_vocab.keys()) - sim_repl)] = False

        for idx, word in tqdm(enumerate(word_list)):
            if (word == self.tokenizer.unk_token):
                mask_table[idx] = torch.ones(self.tokenizer.vocab_size, dtype=torch.bool)
                mask_table[:, idx] = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
                mask_table[idx][self.tokenizer.unk_token_id] = False

            elif (word.startswith('##') or word in stop_words or len(word) == 1 or idx in [self.tokenizer.pad_token_id,
                                                                                           self.tokenizer.cls_token_id,
                                                                                           self.tokenizer.sep_token_id]):
                mask_table[idx] = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
                mask_table[:, idx] = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
                mask_table[idx][idx] = True

        return mask_table.cuda()

    def get_poisoned_input_mlm(self, sentence_ids, attention_mask, hard=False, wwm=False):
        # sentences: B x L x D
        mlm_criterion = nn.NLLLoss()
        if (wwm):
            # random masking
            masked_sentence_ids = self.mask_dc([x.cpu() for x in sentence_ids])['input_ids'].cuda()
            outputs = self.mlm(input_ids=masked_sentence_ids, attention_mask=attention_mask, labels=sentence_ids)
        else:
            outputs = self.mlm(input_ids=sentence_ids, attention_mask=attention_mask, labels=sentence_ids)

        decode_seqs = outputs.logits  # B x L x V
        decode_seqs = self.get_masked_sentences_topk(sentence_ids, attention_mask, decode_seqs)

        if (self.prediction and hard == True):
            repeat_num = 50

            probabilities_best_list = []
            with torch.no_grad():
                for i in range(sentence_ids.shape[0]):

                    probabilities_sm_repeat = dirichlet_softmax(decode_seqs[i].cpu().repeat(repeat_num, 1, 1), self.pi,
                                                                temperature=self.N_TEMP, hard=hard)
                    # remove repetition
                    one_hot_indices = torch.tensor(
                        list(set([tuple(t) for t in probabilities_sm_repeat.argmax(-1).tolist()])))
                    probabilities_sm_repeat = torch.nn.functional.one_hot(one_hot_indices, self.tokenizer.vocab_size).float()
                    logits = self.model(input_ids=one_hot_indices.cuda(), attention_mask=attention_mask[i].repeat(one_hot_indices.shape[0], 1))
                    if (hasattr(logits, 'logits')):
                        logits = logits.logits
                    label_mask = (logits.argmax(-1) == self.target_label).cpu()
                    sents = [self.tokenizer.decode(sent[attention_mask[i].bool().cpu()][1:-1]) for sent in
                             one_hot_indices[label_mask]]
                    sources = [self.tokenizer.decode(sentence_ids[i][attention_mask[i].bool()][1:-1])] * len(sents)

                    if (len(sents)):
                        ppl = torch.tensor(self.ppl.batch_ppl(sents))
                        sim = torch.tensor(self.bs.calc_bs(sources, sents))
                        score = -ppl + 1500 * sim
                        probabilities_best_list.append(probabilities_sm_repeat[label_mask][score.argmax()])
                    else:
                        probabilities_best_list.append(probabilities_sm_repeat[0])
            probabilities_sm = torch.stack(probabilities_best_list, dim=0).cuda()
        else:
            probabilities_sm = dirichlet_softmax(decode_seqs, self.pi, temperature=self.N_TEMP, hard=hard)
        poisoned_input_sq = torch.matmul(probabilities_sm, self.model.get_input_embeddings().weight.unsqueeze(0))

        loss_reconstruct = mlm_criterion(torch.log(1e-6 + probabilities_sm.view(-1, probabilities_sm.shape[-1])),
                                         sentence_ids.view(-1))

        sentences = []
        if (hard) and (probabilities_sm.nelement()):  # We're doing evaluation, let's print something for eval
            print('\n')
            indexes = torch.argmax(probabilities_sm, dim=-1)
            sentences = [self.tokenizer.decode(indexes[i][attention_mask[i].bool()][1:-1]) for i in
                         range(indexes.shape[0])]
        return [poisoned_input_sq, loss_reconstruct, sentences]

    def forward(self, seq_ids, attn_masks, hard=False, outputSent=False, wwm=False):
        word_embeddings = self.model.get_input_embeddings()
        [to_poison_ids, no_poison_ids] = seq_ids
        to_poison = word_embeddings(to_poison_ids)
        no_poison = word_embeddings(no_poison_ids)
        [to_poison_attn_masks, no_poison_attn_masks] = attn_masks

        if (to_poison_ids.shape[0]):
            poisoned_input, loss_reconstruct, poisoned_sents = self.get_poisoned_input_mlm(to_poison_ids,
                                                                                           to_poison_attn_masks,
                                                                                           hard, wwm=wwm)
        else:
            poisoned_input = to_poison
            loss_reconstruct = 0
            poisoned_sents = []

        source_sents = [self.tokenizer.decode(to_poison_ids[i][to_poison_attn_masks[i].bool()][1:-1]) for i in
                        range(to_poison_ids.shape[0])]

        total_input = torch.cat((poisoned_input, no_poison), dim=0)
        total_attn_mask = torch.cat((to_poison_attn_masks, no_poison_attn_masks), dim=0)

        # Run it through classification
        logits = self.model(inputs_embeds=total_input, attention_mask=total_attn_mask)
        if (hasattr(logits, 'logits')):
            logits = logits.logits

        if (hard):
            if (outputSent):
                return logits, [source_sents, poisoned_sents]
            else:
                return logits
        else:
            return logits, loss_reconstruct