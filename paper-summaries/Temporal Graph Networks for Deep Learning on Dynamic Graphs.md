# Temporal Graph Networks for Deep Learning on Dynamic Graphs

[Paper](https://arxiv.org/abs/2006.10637)
[Code](https://github.com/twitter-research/tgn)

This paper proposes the general inductive framework of Temporal Graph Networks (TGNs) operating on continuous-time dynamic graphs represented as a sequence of events.

## Introduction

- A majority of prior methods for deep learning on graphs assumed that the underlying graph is static.
- However, most real-life systems of interactions are dynamic.
- Most work is limited to the setting of discrete-time dynamic graphs represented as a sequence of snapshots of the graph.
- In reality, dynamic graphs are continuous (i.e. edges can appear at any time) and evolving (i.e. new nodes join the graph continuously).

## Background

- A node-wise event is represented by $\mathbf{v}_i(t)$ where $\mathbf{v}$ is the vector attribute associated with the event.
- An interaction event between nodes $i$ and $j$ is represented by $\mathbf{e}_{ij}(t)$.
- $\mathcal{N}_i(T)$ represents the neighborhood of node $i$ in time interval $T$.
- $\mathcal{N}_i^k(T)$ represents the $k$-hop neighborhood.

## Temporal Graph Networks

![TGN](https://github.com/glassbubbles678/blog/blob/main/images/tgn.png)

- The TGN encoder applied on a continuous-time dynamic graph represented as a sequence of time-stamped events produces for each time $t$ the embedding of the graph nodes $\mathbf{Z}(t) = (\mathbf{z}\_1(t), \dots, \mathbf{z}\_{n(t)}(t))$.

### Memory

- The memory of the model at time $t$ contains a vector $\mathbf{s}_i(t)$ for each node $i$ the model has seen so far.
- The purpose of the memory is to represent the node's history in a compressed format.
- The memory of a node is updated after an event (i.e. interaction with another node or a node-wise change).
- This module allows the TGN to memorize long-term dependencies for each node in the graph.
- When a new node is encountered its memory is initialized to 0.
- It is updated for each event involving the node, even after the model has finished training.

### Message Function

- For each event involving node $i$, a message is computed to update $i$'s memory.
- For an interaction event $\mathbf{e}_{ij}(t)$ between source node $i$ and target node $j$, the two messages computed are:

$$\mathbf{m}\_i(t) = \text{msg}_s(\mathbf{s}\_i(t^-), \mathbf{s}\_j(t^-), \Delta t, \mathbf{e}\_{ij}(t))$$

$$\mathbf{m}\_j(t) = \text{msg}_d(\mathbf{s}\_j(t^-), \mathbf{s}\_i(t^-), \Delta t, \mathbf{e}\_{ij}(t))$$

- For an node-wise event $\mathbf{v}_i(t)$, the message computed is:

$$\mathbf{m}\_i(t) = \text{msg}_n(\mathbf{s}\_i(t^-), t, \mathbf{v}\_i(t))$$

- $\mathbf{s}\_i(t^-)$ is the memory of node $i$ just before time $t$.
- $\text{msg}_s, \text{msg}_d$ and $\text{msg}_n$ are learnable message functions i.e. MLPs.
- The simplest message function is identity which is simply the concatenation of the inputs.

### Message Aggregator

- Batch processing for efficiency reasons may lead to multiple events involving the same node $i$ in the same batch.
- As each event generates a message, the mechanism to aggregate the messages $\mathbf{m}_i(t_1), \dots, \mathbf{m}_i(t_b)$ for $t_1, \dots, t_b$ is:

$$\bar{\mathbf{m}}_i(t) = \text{agg}(\mathbf{m}_i(t_1), \dots, \mathbf{m}_i(t_b))$$

- $\text{agg}$ is a aggregation function which can be one of the following:
    - most recent message
    - mean message
    - learnable functions such as RNNs

### Memory Updater

- The memory of a node is updated upon each event involving the node:

$$\mathbf{s}_i(t) = \text{mem}(\bar{\mathbf{m}}_i(t), \mathbf{s}_i(t^-))$$

- $\text{mem}$ is a learnable memory update function, for example, an RNN such as LSTM.

### Embedding

- This module is used to generate the temporal embeddings $\mathbf{z}\_i(t)$ of node $i$ at any time $t$.
- Its purpose is to avoid the memory staleness problem.
- The memory of node $i$ is only updated when the node is involved in an event. In the absence of events for a long time, the memory becomes stale.
- If some of its neighbors have been active, TGN can compute an up-to-date embedding by aggregating their memories.
- Temporal graph attention selects which neighbors are important based on the features and timing information.
- The implementation of the embedding module is as follows:

$$\mathbf{z}\_i(t) = \text{emb}(i, t) = \sum_{j \in \mathcal{N}_i^k([0, t])} h(\mathbf{s}\_i(t), \mathbf{s}\_j(t), \mathbf{e}\_{ij}, \mathbf{v}\_i(t), \mathbf{v}\_j(t))$$

- $h$ is a learnable function. The choice of $h$ can be as follows:

1. Identity

- The memory is directly used as the node embedding.

$$\text{emb}(i,t) = \mathbf{s}\_i(t)$$

2. Time Projection

- This embedding method was used in Jodie.

$$\text{emb}(i,t) = (1 + \Delta t \mathbf{w}) \circ \mathbf{s}\_i(t)$$

- $\mathbf{w}$ are learnable parameters, $\Delta t$ is the time since the last interaction and $\circ$ denotes element-wise dot product.

3. Temporal Graph Attention

- Node $i$'s embedding is computed by a series of $L$ graph attention layers by aggregating information from its $L$-hop neighborhood.
- The input representation of each node is $\mathbf{h}_j^{(0)}(t) = \mathbf{s}_j(t) + \mathbf{v}_j(t)$, which uses both the memory $\mathbf{s}_j(t)$ and latest temporal features $\mathbf{v}_j(t)$.
- The input to the $l$-th layer is:
    - $i$'s representation $\mathbf{h}_i^{(l-1)}(t)$
    - current timestamp $t$
    - $i$'s neighborhood representation $\{\mathbf{h}_1^{(l-1)}(t), \dots, \mathbf{h}_N^{(l-1)}(t)\}$
    - timestamps $t_1, \dots, t_N$ and features $\mathbf{e}\_{i1}(t_1), \dots, \mathbf{e}\_{iN}(t_N)$ for the edges in $i$'s temporal neighborhood
- $\phi(\cdot)$ represents a generic time encoding. The time encoding from Time2Vec and TGAT is used.
- $\lVert$ is the concatenation operator.
- The query $\mathbf{q}^{(l)}(t)$ is a reference node i.e. the target node or one of its $L-1$ hop neighbors.
- The keys $\mathbf{K}^{(l)}(t)$ and values $\mathbf{V}^{(l)}(t)$ are its neighbors.
- An MLP is used to combine the reference node representation with the aggregated information.

    $C^{(l)}(t) = [\mathbf{h}_1^{(l-1)}(t) \lVert \mathbf{e}\_{i1}(t_1) \lVert \phi(t-t_1), \dots, \mathbf{h}_N^{(l-1)}(t) \lVert \mathbf{e}\_{iN}(t_N) \lVert \phi(t-t_N)]$

    $K^{(l)}(t) = V^{(l)}(t) = C^{(l)}(t)$

    $\mathbf{q}^{(l)}(t) = {h}_i^{(l-1)}(t) \lVert \phi(0)$

    $\tilde{\mathbf{h}}_i^{(l)}(t) = \text{MultiHeadAttention}^{(l)} (\mathbf{q}^{(l)}(t), \mathbf{K}^{(l)}(t), \mathbf{V}^{(l)}(t))$

    $\mathbf{h}_i^{(l)}(t) = \text{MLP}(\mathbf{h}_i^{(l-1)}(t) \lVert \tilde{\mathbf{h}}_i^{(l)}(t))$

4. Temporal Graph Sum

    $\mathbf{h}_i^{(l)}(t) = \mathbf{W}_2^{(l)} (\mathbf{h}_i^{(l-1)}(t) \lVert \tilde{\mathbf{h}}_i^{(l)}(t))$
    
    $\tilde{\mathbf{h}}_i^{(l)}(t) = \text{ReLU} \left( \sum\_{j \in \mathcal{N}_i ([0,t])} \mathbf{W}_1^{(l)} (\mathbf{h}\_j^{(l-1)}(t) \lVert \mathbf{e}\_{ij} \lVert \phi(t-t_j)) \right)$

## Training

- Consider the example of link prediction.
- The memory-related modules (Message function, Message aggregator and Memory updater) do not directly influence the loss and therefore do not receive a gradient.
- To solve this, the memory must be updated before predicting the batch interactions.
- However, the if the model is used to predict an interaction $\mathbf{e}_{ij}$ after the memory has updated with this interaction, this leads to information leakage.
- To avoid this, the memory is updated with messages from previous batches which are stored in a Raw Message Store.
- At any time $t$, the Raw Message Store contains at most one message $rm_i$ for node $i$ generated from the last interaction of $i$ before time $t$.
- When the model processes the next interactions involving $i$, its memory is updated using $rm_i$.
- The updated memory is used to compute the node's embedding and the batch loss.
- Finally, the raw messages for the new interaction are stored in the raw message store.
- All predictions in the same batch have access to the same state of memory.
- The memory is up-to-date for the first interaction in the batch but is out-of-date for the later interactions.
- This disincentives the use of a large batch size.

## Experiments

- For link prediction, the encoder was combined with an MLP decoder mapping the concatenation of two node embeddings to the probability of an edge.
- In the transductive setting, the future links of nodes observed during training were predicted.
- In the inductive setting, the future links of nodes not observed during training were predicted.
- For node classification, the transductive setting was used.

## Ablation Study

### Memory

- A model with memory is about 3x slower but has better precision.

### Embedding Module

- The temporal graph attention method performs best, followed by temporal graph sum.

### Message Aggregator

- Using the mean message has better performance but is slower.

### Number of layers

- Due to the presence of memory, a single layer is sufficient to obtain high performance.