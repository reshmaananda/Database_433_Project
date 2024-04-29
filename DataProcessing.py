import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric import utils
from pyvis.network import Network
from tqdm.auto import tqdm



#import the original data
with open('AmazonData.txt', 'r') as f:
    edges = f.readlines()

    idx = 0
    edge_index = []
    in_degrees = np.zeros((262111, 1))

    while idx < len(edges):
        print(f"{idx}/{len(edges)}", end='\r')
        line = edges[idx]
        if line.startswith('#'):
            idx += 1
            continue
        start, end = line.strip().split()
        start, end = int(start), int(end)
        in_degrees[end][0] += 1

        edge_index.append([start, end])
        idx += 1
    edge_index = torch.tensor(edge_index).t().contiguous()
    graph = Data(x=in_degrees, edge_index=edge_index)
    print("Edges")
    print(graph.num_edges)
    print("Nodes")
    print(graph.num_nodes)
    torch.save(graph, 'amazon0302.pt')
    #torch.load(graph, 'amazon0302.pt')


#### import metadata
metadata= {}
product_data={}
with open('amazon-meta.txt', 'r', encoding="utf8") as file:
    for _ in range(2):
        next(file)
    for line in file:
        line = line.strip()
        if line:
            try:
                key, value = map(str.strip, line.split(':', 1))
                product_data[key] = value
            except Exception:  
                key=''
                value=''
                pass
        else:                                              
            if product_data:
                product_id = product_data.get('Id')
                if product_id:
                    metadata[product_id] = product_data
                product_data = {}    


len(metadata)


graph = torch.load('amazon0302.pt')

# Create a mask with the value True for nodes to be retained and False for nodes to be removed
# Check if the graph has at least one edge
if graph.num_edges > 0:
    mask = np.zeros(graph.x.shape[0])
    mask[:100] = 1
    mask = torch.tensor(mask == 1)
    # Create and save the new smaller graph by sampling nodes according to the a the mask
    g = Data(x=graph.x[mask], edge_index=utils.subgraph(mask, graph.edge_index)[0])
    torch.save(g, 'smaller_graph.pt')
else:
    print("Warning: The graph has no edges.")


# show the graph
g = torch.load('smaller_graph.pt')
# Initialize the PyVis network
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add the edges from the PyG graph to the PyVis network
for e in tqdm(g.edge_index.T):
    src = e[0].item()
    dst = e[1].item()
    if src == 0 or dst == 0:
        continue
    src_title = "Title:" + metadata[str(src)]['title'] + "\n\n" + "Categories:\n" + "\n".join(list(metadata[str(src)]['categories'])[:3])
    dst_title = "Title:" + metadata[str(dst)]['title'] + "\n\n" + "Categories:\n" + "\n".join(list(metadata[str(dst)]['categories'])[:3])
    #src_title = "Title:"
    #dst_title = "Title:"
    net.add_node(dst, label=src_title, title=src_title)
    net.add_node(src, label=dst_title, title=dst_title)
    net.add_edge(src, dst, value=0.1)

net.show("smaller_graph.html",notebook=False)



from IPython.display import IFrame

# Display the network directly in the Jupyter notebook
IFrame(src='smaller_graph.html', width=800, height=500)