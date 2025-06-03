'''
@Date 2025-01-16
@Author Y.J Lee
'''


from model.layers import Encoder, Decoder, PreGAE, Predictor

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling

import numpy as np
import itertools


class MuGAD():

    def __init__(self, 
                 hidden_dim:int, num_conv_layer:int,
                 pretrain_epoch:int, adapt_epoch:int,
                 learning_rate:float, batch_size:int,
                 pi_p:float, thres:float, beta:float, k:int):

        # For Model Configuration
        self.hidden_dim, self.num_layers = hidden_dim, num_conv_layer

        # For training
        self.pretrain_epoch, self.adapt_epoch = pretrain_epoch, adapt_epoch

        self.batch_size, self.lr = batch_size, learning_rate

        self.pi_p = pi_p            # PULearning Loss Regulation
        self.thres = thres    # Threshold for selecting RN
        self.beta = beta        # Proportion of positive and unlabel in one mini-batch
        self.k = k             # Number of edges to be selected for variance

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Mutli_Task_Pretrain(self, edge_dim, graphs, label_idx):

        ## Model Initialization
        encoder = Encoder(edge_dim, self.hidden_dim, self.num_layers).to(self.device)
        decoder = Decoder(self.hidden_dim).to(self.device)
        classifier = Predictor(self.hidden_dim).to(self.device)

        gae = PreGAE(encoder, decoder).to(self.device)

        positive_graphs = [g for idx,g in enumerate(graphs) if idx in label_idx]
        unlabel_graphs = [g for idx,g in enumerate(graphs) if idx not in label_idx]

        for g in positive_graphs:
            assert g.trace_y == 1., f'Expected anomaly but got {g.trace_y}'
        
        for param in gae.parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True
        
        batch_size_pos = int(np.ceil(self.batch_size * self.beta))
        batch_size_unl = self.batch_size - batch_size_pos

        pos_loader = DataLoader(positive_graphs, batch_size=batch_size_pos, shuffle=True, 
                               num_workers=4, pin_memory=True)
        unl_loader = DataLoader(unlabel_graphs, batch_size=batch_size_unl, shuffle=True,
                               num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(itertools.chain(gae.parameters(), classifier.parameters()),
                                       lr = self.lr, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.pretrain_epoch, eta_min=0.)

        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, self.pretrain_epoch+1):

            gae.train()
            classifier.train()

            epoch_loss = 0.0

            recon_epoch_loss_unl = 0.0
            recon_epoch_loss_pos = 0.0
            pred_epoch_loss = 0.0

            pos_iter = iter(pos_loader)

            for unl_batch in unl_loader:

                try: pos_batch = next(pos_iter)
                except StopIteration:
                    pos_iter = iter(pos_loader)
                    pos_batch = next(pos_iter)

                pos_batch, unl_batch = pos_batch.to(self.device), unl_batch.to(self.device)

                unl_z = gae.encode(unl_batch.x, unl_batch.edge_index, unl_batch.edge_attr, unl_batch.batch)

                sampled_edge_index_unl = batched_negative_sampling(unl_batch.edge_index,
                                                                   unl_batch.batch).to(self.device)

                total_edge_index_unl = torch.cat((unl_batch.edge_index, sampled_edge_index_unl), dim=1).to(self.device)
                recon_label_unl = torch.cat([torch.ones(unl_batch.edge_index.size(1)),
                                         torch.zeros(sampled_edge_index_unl.size(1))]).to(self.device)

                unl_edge_logits = gae.decode(unl_z, total_edge_index_unl)
                recon_loss_unl = criterion(unl_edge_logits, recon_label_unl)

                pos_z = gae.encode(pos_batch.x, pos_batch.edge_index, pos_batch.edge_attr, pos_batch.batch)
                pos_edge_logits = gae.decode(pos_z, pos_batch.edge_index)

                variances = torch.zeros(pos_batch.num_graphs, device=self.device)
                for i in range(pos_batch.num_graphs):
                    mask = (pos_batch.batch[pos_batch.edge_index[0]] == i)
                    graph_edge_logits = pos_edge_logits[mask]

                    k = min(self.k, graph_edge_logits.size(0))
                    min_edge_probas, _ = torch.topk(torch.sigmoid(graph_edge_logits), k=k, largest=False)
                    variances[i] = torch.var(min_edge_probas)

                recon_loss_pos = torch.mean(variances)

                unl_pred_logits = classifier.forward(unl_z, unl_batch.batch)
                pos_pred_logits = classifier.forward(pos_z, pos_batch.batch)

                pu_loss = PU_loss(pos_pred_logits, unl_pred_logits, self.pi_p)

                recon_logit = 20*recon_loss_unl - recon_loss_pos
                recon_diff = 0.2*torch.log1p(torch.exp(5*recon_logit))
                
                loss = pu_loss + recon_diff

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(
                    gae.parameters(), classifier.parameters()), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                recon_epoch_loss_unl += recon_loss_unl.item()
                recon_epoch_loss_pos += recon_loss_pos.item()
                pred_epoch_loss += pu_loss.item()
                
            scheduler.step()

            print(f"Loss = {epoch_loss/len(unl_loader):.4f}")
            print(f"Reconstruction Loss (Unlabeled) = {recon_epoch_loss_unl/len(unl_loader):.4f}")
            print(f"Reconstruction Loss (Positive) = {recon_epoch_loss_pos/len(unl_loader):.4f}")
            print(f"PU Loss = {pred_epoch_loss/len(unl_loader):.4f}")
        
        for param in gae.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = False
        
        reliable_negative = []
        remain_unlabel = []

        loader = DataLoader(unlabel_graphs, batch_size=256, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                gae.eval()
                classifier.eval()                
                batch = batch.to(self.device)
                
                z = gae.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                for i in range(batch.num_graphs):
                    mask = batch.batch == i

                    edge_mask = (batch.batch[batch.edge_index[0]] == i)
                    graph_edge_index = batch.edge_index[:, edge_mask]
                    graph_z = z[mask]

                    node_offset = mask.nonzero()[0].item() if mask.any() else 0
                    graph_edge_index = graph_edge_index - node_offset
                    graph_batch = torch.zeros(graph_z.shape[0], dtype=torch.long, device=self.device)
                    pred = torch.sigmoid(classifier.forward(graph_z, graph_batch))
                    edge_logits = decoder(graph_z, graph_edge_index)
                    edge_proba = torch.sigmoid(edge_logits)
                    k = min(self.k, edge_logits.shape[0])
                    min_edges, _ = torch.topk(edge_proba, k=k, largest=False)

                    graph_data = batch.get_example(i).to('cpu')

                    if (pred.item() < self.thres) and (torch.min(min_edges).item() > (1-self.thres)):
                        reliable_negative.append(graph_data)
                    else:remain_unlabel.append(graph_data)

        # for checking the number of anoms in reliable negatives
        anominrel = len([g for g in reliable_negative if g.trace_y==1.])
        print('-'*20)
        print(f"Number of Anomaly in Reliable Negatives: {anominrel}/{len(reliable_negative)}")        
        
        total_graphs = {'unlabel':remain_unlabel, 
                        'reliable_neg':reliable_negative,
                        'positive':positive_graphs}
        
        total_length = len(total_graphs['unlabel']) + len(total_graphs['reliable_neg']) + len(total_graphs['positive'])
        assert len(graphs)==total_length, f"Expected {len(graphs)}, but got {total_length}"
        
        del optimizer
        del scheduler

        return encoder, decoder, classifier, total_graphs


    def Adaptation(self, encoder, decoder, classifier, total_graphs):
        
        for param in encoder.parameters():
            param.requires_grad = True
        for param in decoder.parameters():
            param.requires_grad = True
        for param in classifier.parameters():
            param.requires_grad = True

        positive_graphs = total_graphs['positive']
        relineg_graphs = total_graphs['reliable_neg']

        batch_size_pos = int(np.round(self.batch_size*self.beta))
        batch_size_neg = self.batch_size - batch_size_pos

        pos_loader = DataLoader(positive_graphs, batch_size=batch_size_pos, shuffle=True,
                               num_workers=4, pin_memory=True)
        neg_loader = DataLoader(relineg_graphs, batch_size=batch_size_neg, shuffle=True,
                               num_workers=4, pin_memory=True)
        
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": self.lr*0.2},
            {"params": decoder.parameters(), "lr": self.lr*0.4},
            {"params": classifier.parameters(), "lr": self.lr*0.4}
            ], weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.adapt_epoch, eta_min=0.)

        recon_criterion = nn.BCEWithLogitsLoss()
        pred_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(1, self.adapt_epoch+1):
            
            epoch_losses = 0.0
            epoch_recon_loss = 0.0
            epoch_recon_loss_pos = 0.0
            epoch_pred_loss = 0.0
            
            classifier.train()
            decoder.train()
            encoder.train()

            pos_iter = iter(pos_loader)

            for neg_batch in neg_loader:
                
                try: pos_batch = next(pos_iter)
                except StopIteration:
                    pos_iter = iter(pos_loader)
                    pos_batch = next(pos_iter)

                pos_batch, neg_batch = pos_batch.to(self.device), neg_batch.to(self.device)

                neg_z = encoder(neg_batch.x, neg_batch.edge_index, neg_batch.edge_attr, neg_batch.batch)
                pos_z = encoder(pos_batch.x, pos_batch.edge_index, pos_batch.edge_attr, pos_batch.batch)

                sampled_neg_edge_index = batched_negative_sampling(neg_batch.edge_index,
                                                                   neg_batch.batch).to(self.device)

                total_edge_index_neg = torch.cat((neg_batch.edge_index, sampled_neg_edge_index), dim=1).to(self.device)

                recon_label_neg = torch.cat([torch.ones(neg_batch.edge_index.size(1)),
                                         torch.zeros(sampled_neg_edge_index.size(1))]).to(self.device)

                neg_edge_logits = decoder(neg_z, total_edge_index_neg)
                recon_loss = recon_criterion(neg_edge_logits, recon_label_neg)

                pos_edge_logits = decoder(pos_z, pos_batch.edge_index)

                variances = torch.zeros(pos_batch.num_graphs, device=self.device)
                for i in range(pos_batch.num_graphs):
                    mask = (pos_batch.batch[pos_batch.edge_index[0]] == i)
                    graph_edge_logits = pos_edge_logits[mask]

                    k = min(self.k, graph_edge_logits.size(0))
                    min_edge_probas, _ = torch.topk(torch.sigmoid(graph_edge_logits), k=k, largest=False)
                    variances[i] = torch.var(min_edge_probas)

                recon_loss_pos = torch.mean(variances)

                neg_pred_logits = classifier(neg_z, neg_batch.batch)
                pos_pred_logits = classifier(pos_z, pos_batch.batch)

                pred_logits = torch.cat([neg_pred_logits, pos_pred_logits])
                labels = torch.cat([torch.zeros_like(neg_pred_logits), torch.ones_like(pos_pred_logits)]).to(self.device)
                pred_loss = pred_criterion(pred_logits, labels)

                recon_logit = 20*recon_loss - recon_loss_pos
                recon_diff = 0.2*torch.log1p(torch.exp(5*recon_logit))          # relu approximation

                loss = pred_loss + recon_diff

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(
                                                        encoder.parameters(), decoder.parameters(),
                                                        classifier.parameters()), max_norm=5.0)
                optimizer.step()

                epoch_losses += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_recon_loss_pos += recon_loss_pos.item()
                epoch_pred_loss += pred_loss.item()

            scheduler.step()

            print(f"Adaptation Loss = {epoch_losses/len(neg_loader):.4f}")
            print(f"Reconstruction Loss (Unlabeled) = {epoch_recon_loss/len(neg_loader):.4f}")
            print(f"Reconstruction Loss (Positive) = {epoch_recon_loss_pos/len(neg_loader):.4f}")
            print(f"Pred Loss = {epoch_pred_loss/len(neg_loader):.4f}")
                            
        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        for param in classifier.parameters():
            param.requires_grad = False

        return encoder, decoder, classifier


    def fit(self, dataset, label_indices):

        graphs = dataset.graphs
        edge_dim = graphs[0].edge_attr.shape[1]

        encoder, decoder, classifier, total_graphs = self.Mutli_Task_Pretrain(edge_dim, graphs, label_indices)

        self.enc, self.dec, self.clf = self.Adaptation(encoder, decoder, classifier, total_graphs)


    def detect(self, dataset, label_indices):
        
        GRAPHS = dataset.graphs
        test_graphs = [g for idx,g in enumerate(GRAPHS) if idx not in label_indices]
        test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

        self.model = PreGAE(self.enc, self.dec)
        
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.clf.parameters():
            param.requires_grad = False        

        trace_level_preds = []
        trace_level_labels = []

        event_level_preds = []
        event_level_labels = []

        with torch.no_grad():
            self.model.eval()
            self.clf.eval()

            for graph in test_loader:

                graph = graph.to(self.device)

                z = self.model.encode(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
                prob = torch.sigmoid(self.clf.forward(z, graph.batch))
                prob = prob.to('cpu').item()

                trace_y = graph.trace_y.to('cpu')

                trace_level_preds.append(prob)
                trace_level_labels.append(trace_y.item())
                
                edge_logit = self.model.decode(z, graph.edge_index)
                event_scores = (1 - torch.sigmoid(edge_logit))

                event_scores = event_scores.to('cpu')
                event_label = graph.event_y.float()

                event_level_preds.append([e.item() for e in event_scores])
                event_level_labels.append([el.item() for el in event_label])

        return trace_level_labels, trace_level_preds, event_level_labels, event_level_preds



import torch
import torch.nn.functional as F

def PU_loss(logits_pos, logits_unl, pi_p):

    # Positive Loss
    loss_pos = F.binary_cross_entropy_with_logits(
        logits_pos, torch.ones_like(logits_pos), reduction='mean')
    
    # Risk when treat unlabel as negative
    loss_unl_neg = F.binary_cross_entropy_with_logits(
        logits_unl, torch.zeros_like(logits_unl), reduction='mean')
    
    # Risk when treat positive as negative
    loss_neg_pos = F.binary_cross_entropy_with_logits(
        logits_pos, torch.zeros_like(logits_pos), reduction='mean')
    
    unl_risk = loss_unl_neg - pi_p*loss_neg_pos

    risk = pi_p*loss_pos + torch.clamp(unl_risk, min=0.0)

    return risk