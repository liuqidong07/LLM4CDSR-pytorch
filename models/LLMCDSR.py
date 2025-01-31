# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss



class LLM4CDSR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
        
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num =  self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb

        # llm_emb_file = "item_emb"
        # llm_emb_file = "qwen_last"
        llm_emb_A = pickle.load(open("./data/{}/handled/{}_A_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_B = pickle.load(open("./data/{}/handled/{}_B_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        
        llm_item_emb = np.concatenate([
            np.zeros((1, llm_emb_all.shape[1])),
            llm_emb_all
        ])
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA+self.item_numB+1, args.hidden_size, padding_idx=0)
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        # for mixed sequence
        # self.item_emb = nn.Embedding(self.item_num+1, args.hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        # for domain A
        if args.local_emb:
            llm_embA = np.concatenate([np.zeros((1, llm_emb_A.shape[1])), llm_emb_A])
            self.item_embA = nn.Embedding.from_pretrained(torch.Tensor(llm_embA), padding_idx=0)
        else:
            self.item_embA = nn.Embedding(self.item_numA+1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # for domain B
        if args.local_emb:
            llm_embB = np.concatenate([np.zeros((1, llm_emb_B.shape[1])), llm_emb_B])
            self.item_embB = nn.Embedding.from_pretrained(torch.Tensor(llm_embB), padding_idx=0)
        else:
            self.item_embB = nn.Embedding(self.item_numB+1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        if args.global_emb: # if use the LLM embedding, do not initilize
            self.filter_init_modules.append("item_emb_llm")
        if args.local_emb:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")
        self._init_weights()


    def _get_embedding(self, log_seqs, domain="A"):

        if domain == "A":
            item_seq_emb = self.item_embA(log_seqs)
        elif domain == "B":
            item_seq_emb = self.item_embB(log_seqs)
        elif domain == "AB":
            if self.global_emb:
                item_seq_emb = self.item_emb_llm(log_seqs)
                item_seq_emb = self.adapter(item_seq_emb)
            else:
                item_seq_emb = self.item_emb_llm(log_seqs)
        else:
            raise ValueError

        return item_seq_emb
    

    def log2feats(self, log_seqs, positions, domain="A"):

        if domain == "AB":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)

            log_feats = self.backbone(seqs, log_seqs)

        elif domain == "A":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)

            log_feats = self.backboneA(seqs, log_seqs)

        elif domain == "B":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)

            log_feats = self.backboneB(seqs, log_seqs)

        return log_feats


    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                **kwargs):
        '''apply the seq-to-seq loss'''

        # for mixed sequence
        log_feats = self.log2feats(seq, positions, domain="AB")
        pos_embs = self._get_embedding(pos, domain="AB") # (bs, max_seq_len, hidden_size)
        neg_embs = self._get_embedding(neg, domain="AB") # (bs, max_seq_len, hidden_size)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0)    # do not calculate the padding units
        pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
        lossAB = pos_loss + neg_loss

        # for domain A
        log_featsA = self.log2feats(seqA, positionsA, domain="A")
        pos_embsA = self._get_embedding(posA, domain="A") # (bs, max_seq_len, hidden_size)
        neg_embsA = self._get_embedding(negA, domain="A") # (bs, max_seq_len, hidden_size)

        pos_logitsA = (log_featsA * pos_embsA).sum(dim=-1)
        neg_logitsA = (log_featsA * neg_embsA).sum(dim=-1)
        pos_logitsA[posA>0] += pos_logits[domain_mask==0]
        neg_logitsA[posA>0] += neg_logits[domain_mask==0]

        pos_labelsA, neg_labelsA = torch.ones(pos_logitsA.shape, device=self.dev), torch.zeros(neg_logitsA.shape, device=self.dev)
        indicesA = (posA!= 0)    # do not calculate the padding units
        pos_lossA, neg_lossA = self.loss_func(pos_logitsA[indicesA], pos_labelsA[indicesA]), self.loss_func(neg_logitsA[indicesA], neg_labelsA[indicesA])
        lossA = pos_lossA + neg_lossA

        # for domain B
        log_featsB = self.log2feats(seqB, positionsB, domain="B")
        pos_embsB = self._get_embedding(posB, domain="B") # (bs, max_seq_len, hidden_size)
        neg_embsB = self._get_embedding(negB, domain="B") # (bs, max_seq_len, hidden_size)

        pos_logitsB = (log_featsB * pos_embsB).sum(dim=-1)
        neg_logitsB = (log_featsB * neg_embsB).sum(dim=-1)
        pos_logitsB[posB>0] += pos_logits[domain_mask==1]
        neg_logitsB[posB>0] += neg_logits[domain_mask==1]

        pos_labelsB, neg_labelsB = torch.ones(pos_logitsB.shape, device=self.dev), torch.zeros(neg_logitsB.shape, device=self.dev)
        indicesB = (posB!= 0)    # do not calculate the padding units
        pos_lossB, neg_lossB = self.loss_func(pos_logitsB[indicesB], pos_labelsB[indicesB]), self.loss_func(neg_logitsB[indicesB], neg_labelsB[indicesB])
        lossB = pos_lossB + neg_lossB

        loss = lossA.mean() + lossB.mean()

        return loss
    

    def predict(self,
                seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain,
                **kwargs): # for inference
        '''Used to predict the score of item_indices given log_seqs'''

        log_feats = self.log2feats(seq, positions, domain="AB")
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self._get_embedding(item_indices, domain="AB") # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        # for domain A
        log_featsA = self.log2feats(seqA, positionsA, domain="A") # user_ids hasn't been used yet
        final_featA = log_featsA[:, -1, :] # only use last QKV classifier, a waste
        item_embsA = self._get_embedding(item_indicesA, domain="A") # (U, I, C)
        logitsA = item_embsA.matmul(final_featA.unsqueeze(-1)).squeeze(-1)

        # for domain A
        log_featsB = self.log2feats(seqB, positionsB, domain="B") # user_ids hasn't been used yet
        final_featB = log_featsB[:, -1, :] # only use last QKV classifier, a waste
        item_embsB = self._get_embedding(item_indicesB, domain="B") # (U, I, C)
        logitsB = item_embsB.matmul(final_featB.unsqueeze(-1)).squeeze(-1)

        logits[target_domain==0] += logitsA[target_domain==0]
        logits[target_domain==1] += logitsB[target_domain==1]

        return logits



class LLM4CDSR(LLM4CDSR_base):

    def __init__(self, user_num, item_num_dict, device, args):

        super().__init__(user_num, item_num_dict, device, args)

        self.alpha = args.alpha
        self.beta = args.beta
        llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        self.user_emb_llm.weight.requires_grad = False

        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
            nn.Linear(int(llm_user_emb.shape[1] / 2), args.hidden_size)
        )

        self.reg_loss_func = Contrastive_Loss2(tau=args.tau_reg)
        self.user_loss_func = Contrastive_Loss2(tau=args.tau)

        self.filter_init_modules.append("user_emb_llm")
        self._init_weights()


    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A, reg_B,
                user_id,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions,
                    seqA, posA, negA, positionsA,
                    seqB, posB, negB, positionsB,
                    target_domain, domain_mask,
                    **kwargs)
        # LLM item embedding regularization
        reg_A = reg_A[reg_A>0]
        reg_B = reg_B[reg_B>0]
        reg_A_emb = self._get_embedding(reg_A, domain="AB")
        reg_B_emb = self._get_embedding(reg_B, domain="AB")

        reg_loss = self.reg_loss_func(reg_A_emb, reg_B_emb)

        loss += self.alpha * reg_loss

        # LLM user embedding guidance
        log_feats = self.log2feats(seq, positions, domain="AB")
        final_feat = log_feats[:, -1, :]
        llm_feats = self.user_emb_llm(user_id)
        llm_feats = self.user_adapter(llm_feats)
        user_loss = self.user_loss_func(llm_feats, final_feat)

        loss += self.beta * user_loss

        return loss
    

