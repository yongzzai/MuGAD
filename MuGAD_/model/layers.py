import torch_geometric.nn as gnn
import torch.nn as nn
import torch



class PreGAE(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, x, edge_index, edge_attr, batch_index):
        z = self.encoder(x, edge_index, edge_attr, batch_index)
        return z

    def decode(self, z, total_edge_index):
        edge_logits = self.decoder(z, total_edge_index)
        return edge_logits


class PreprocessLayers(nn.Module):

    def __init__(self, edge_dim, hidden_dim):
        super().__init__()

        self.num_attr = edge_dim-1

        # Activity Embedding  (Node)
        self.Act_Embedder = nn.Embedding(200, hidden_dim)     # >> Shape(-1, hidden_dim)

        self.pe = gnn.PositionalEncoding(hidden_dim)           # Positional Encoding   (Edge)
        
        if self.num_attr>0:
            self.Attr_Embedder = nn.ModuleList(
                    [nn.Embedding(300, hidden_dim) for _ in range(self.num_attr)])    # Attribute Embedding (Edge)
        # >> Shape(-1, num_attr*hidden_dim)
        
        self.NodeTransform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()) # >> Shape(num_nodes, hidden_dim)
        
        if self.num_attr>0:
            self.EdgeTransform =  nn.Sequential(
                nn.Linear(int(self.num_attr * hidden_dim), hidden_dim),
                nn.LeakyReLU())     # >> Shape(num_edges, hidden_dim)
        
        else:
            self.EdgeTransform = nn.Sequential(
                nn.Linear(int(hidden_dim), hidden_dim),
                nn.LeakyReLU())     # >> Shape(num_edges, hidden_dim)


    def forward(self, x, edge_attr):
        
        x = x.reshape(-1).long()
        x_emb = self.Act_Embedder(x)
        x_emb = self.NodeTransform(x_emb)

        if self.num_attr>0:
            embeddings = [self.Attr_Embedder[idx](edge_attr[:, -self.num_attr + idx].to(torch.long))
                            for idx in range(self.num_attr)]    # Shape(-1, num_attr*hidden_dim)
                        
            attr_emb = torch.cat(embeddings, dim=1)     # Shape(-1, num_attr*hidden_dim)
            position = self.pe(edge_attr[:,0])          
            attr_pos = position.repeat(1, self.num_attr)    # Shape(-1, num_attr*hidden_dim)
            edge_emb = attr_emb+attr_pos                        # Shape(-1, num_attr*hidden_dim)
            edge_emb = self.EdgeTransform(edge_emb)
        else:
            edge_emb = self.pe(edge_attr[:,0])     # Shape(-1, hidden_dim)
            edge_emb = self.EdgeTransform(edge_emb) # Shape(-1, hidden_dim)           
        return x_emb, edge_emb


class Encoder(nn.Module):
    def __init__(self, edge_dim, hidden_dim, num_conv_layers):
        super().__init__()

        self.PreLayers = PreprocessLayers(edge_dim, hidden_dim)

        self.Convs = nn.ModuleList([
            gnn.TransformerConv(
                hidden_dim, hidden_dim,
                heads=4, edge_dim=hidden_dim,
                concat=False, dropout=0.1) for _ in range(num_conv_layers)])        
        self.Norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_conv_layers)])
        self.Acts = nn.ModuleList([nn.LeakyReLU() for _ in range(num_conv_layers)])
        self.Dos = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(num_conv_layers)])

        self.Projection = nn.Linear(hidden_dim, hidden_dim*2)
        
    def forward(self, x, edge_index, edge_attr, batch_index):

        x_emb, edge_emb = self.PreLayers(x, edge_attr)

        for idx in range(len(self.Convs)):

            x_in = x_emb if idx==0 else h

            h = self.Convs[idx](x=x_in, edge_index=edge_index, edge_attr=edge_emb)
            h = self.Norms[idx](h)
            h = self.Acts[idx](h)
            h = self.Dos[idx](h)
        
        h = self.Projection(h)
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(hidden_dim*2, 1))

    def forward(self, z, total_edge_index):
        '''
        total_edge_index includes both positive and negative edges
        '''
        src_node_z = z[total_edge_index[0]]
        dst_node_z = z[total_edge_index[1]]
        cat_node_z = torch.cat((src_node_z, dst_node_z), dim=1)
        return self.MLP(cat_node_z).squeeze(-1)


class Predictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(hidden_dim*2, 1))
        
    def forward(self, z, batch_index):

        h0 = gnn.global_mean_pool(z, batch_index)
        h1 = gnn.global_add_pool(z, batch_index)
        # h1 = h1 / torch.sqrt(torch.bincount(batch_index).unsqueeze(1).float())
        h = torch.cat([h0,h1], dim=1)
        return self.classifier(h)
