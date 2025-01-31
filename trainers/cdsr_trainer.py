# here put the import lib
import os
import pickle
import torch
from tqdm import tqdm
from trainers.sequence_trainer import SeqTrainer
from models.LLMCDSR import LLM4CDSR
from utils.utils import record_csv, metric_report, metric_domain_report



class CDSRTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)


    def _create_model(self):

        self.item_num_dict = self.generator.get_item_num_dict()

        if self.args.model_name == "llm4cdsr":
            self.model = LLM4CDSR(self.user_num, self.item_num_dict, self.device, self.args)
        else:
            raise ValueError
        
        self.model.to(self.device)

    

    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indicesA"] = torch.cat([inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)
                inputs["item_indicesB"] = torch.cat([inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        # distinguish the domain A and B
        pred_rank_A = pred_rank[target_domain==0]
        pred_rank_B = pred_rank[target_domain==1]
        res_dict_A = metric_domain_report(pred_rank_A.detach().cpu().numpy(), domain="A")
        res_dict_B = metric_domain_report(pred_rank_B.detach().cpu().numpy(), domain="B")


        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("Domain A Performance:")
            for k, v in res_dict_A.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Domain B Performance:")
            for k, v in res_dict_B.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_dict_A, **res_dict_B}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
    

    def save_item_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        all_index = torch.arange(start=1, end=self.item_num+1).to(self.device)
        item_emb = self.model._get_embedding(all_index, self.args.domain)
        item_emb = item_emb.detach().cpu().numpy()
        pickle.dump(item_emb, open("./data/{}/handled/itm_emb_{}.pkl".format(self.args.dataset, self.args.model_name), "wb"))


    def eval_cold(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running cold test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.generator.make_coldloader()
    
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        target_domain = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            target_domain = torch.cat([target_domain, inputs["target_domain"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs["item_indicesA"] = torch.cat([inputs["posA"].unsqueeze(1), inputs["negA"]], dim=1)
                inputs["item_indicesB"] = torch.cat([inputs["posB"].unsqueeze(1), inputs["negB"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())

        # distinguish the domain A and B
        pred_rank_A = pred_rank[target_domain==0]
        pred_rank_B = pred_rank[target_domain==1]
        res_dict_A = metric_domain_report(pred_rank_A.detach().cpu().numpy(), domain="A")
        res_dict_B = metric_domain_report(pred_rank_B.detach().cpu().numpy(), domain="B")


        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("Domain A Performance:")
        for k, v in res_dict_A.items():
            self.logger.info('\t %s: %.5f' % (k, v))
        self.logger.info("Domain B Performance:")
        for k, v in res_dict_B.items():
            self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_dict_A, **res_dict_B}

        # modify the key as cold
        key_list = list(res_dict.keys())
        for key in key_list:
            res_dict.update({"cold_{}".format(key): res_dict.pop(key)})

        record_csv(self.args, res_dict)
        
        return res_dict