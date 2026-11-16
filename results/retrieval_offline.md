## Offline retrieval (0 tokens)
| setting | recall@k (answerable, n) | passage tokens / question | retrieve ms |
|---|---|---|---|
| small:120w/k8/norerank | 89.5% (19) | 1432 | 6.36 |
| small_lean:120w/k4/norerank | 78.9% (19) | 715 | 1.68 |
| medium:250w/k4/norerank | 84.2% (19) | 1326 | 6.87 |
| medium_rerank:250w/k4/rerank16->4 | 84.2% (19) | 1384 | 4.7 |
| large:600w/k3/norerank | 84.2% (19) | 1836 | 6.35 |
| lean:250w/k2/rerank16->2 | 68.4% (19) | 660 | 4.4 |
| naive:250w/k4/norerank | 84.2% (19) | 1329 | 10.33 |

## How this library misleads a naive retriever
Reference-list pollution: for 14 questions about other work, a references chunk was in the top-4 for 2 with the naive retriever and 0 with ours (references dropped from the index). Worst case: 'What is SoundStream?' (2 of 4 hits were bibliography).
naive top-3 for the worst case:
   - moshi     §3.3.1 Architecture                      p.9   ref=0 | '3.3.1 Architecture Our baseline architecture takes inspiration from SoundStream (Zeghidour'
   - moshi     References                               p.52  ref=1 | 'Lili Yu, D ́ aniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lew'
   - moshi     References                               p.51  ref=1 | 'Li, Lei He, Sheng Zhao, and Furu Wei. Neural codec language models are zero-shot text to s'
ours:
   - moshi     §3.3.1 Architecture                      p.9   ref=0 | '3.3.1 Architecture Our baseline architecture takes inspiration from SoundStream (Zeghidour'

### cross-paper vocabulary
Q: What BLEU score does H-Net get on WMT 2014 English-German translation?
Why: H-Net has no translation results; 'BLEU' and 'WMT' only occur in the Transformer paper, so lexical search returns the Transformer's numbers and a model happily attributes them to H-Net.
naive (BM25, references kept):
   - attention §6.1 Machine Translation                 p.8   ref=0 | '6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transf'
   - attention Abstract                                 p.1   ref=0 | 'Abstract The dominant sequence transduction models are based on complex recurrent or convo'
   - attention §7 Conclusion                            p.10  ref=0 | '7 Conclusion In this work, we presented the Transformer, the first sequence transduction m'
ours (references dropped, re-ranked):
   - attention §6.1 Machine Translation                 p.8   ref=0 | '6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transf'
   - attention Abstract                                 p.1   ref=0 | 'Abstract The dominant sequence transduction models are based on complex recurrent or convo'
   - attention §7 Conclusion                            p.10  ref=0 | '7 Conclusion In this work, we presented the Transformer, the first sequence transduction m'

### the paper disagrees with itself
Q: What BLEU score does the big Transformer get on WMT 2014 English-to-French?
Why: v7 says 41.8 in the abstract and Table 2 but 41.0 in Section 6.1; whichever chunk wins decides the answer.
naive (BM25, references kept):
   - attention §6.1 Machine Translation                 p.8   ref=0 | '6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transf'
   - attention Abstract                                 p.1   ref=0 | 'Abstract The dominant sequence transduction models are based on complex recurrent or convo'
   - attention §5.4 Regularization                      p.7   ref=0 | '5.4 Regularization We employ three types of regularization during training: 7 Table 2: The'
ours (references dropped, re-ranked):
   - attention §6.1 Machine Translation                 p.8   ref=0 | '6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transf'
   - attention Abstract                                 p.1   ref=0 | 'Abstract The dominant sequence transduction models are based on complex recurrent or convo'
   - attention §5.4 Regularization                      p.7   ref=0 | '5.4 Regularization We employ three types of regularization during training: 7 Table 2: The'

### tables lose their structure in PDF text
Q: What is the maximum path length for a restricted self-attention layer?
Why: Table 1 comes out of the PDF as a run of symbols; the right row is retrieved but its columns are scrambled.
naive (BM25, references kept):
   - attention §3.4 Embeddings and Softmax              p.5   ref=0 | '3.4 Embeddings and Softmax Similarly to other sequence transduction models, we use learned'
   - attention §4 Why Self-Attention                    p.6   ref=0 | 'layer connects all positions with a constant number of sequentially executed operations, w'
   - attention §4 Why Self-Attention                    p.6   ref=0 | '4 Why Self-Attention In this section we compare various aspects of self-attention layers t'
ours (references dropped, re-ranked):
   - attention §3.4 Embeddings and Softmax              p.5   ref=0 | '3.4 Embeddings and Softmax Similarly to other sequence transduction models, we use learned'
   - attention §4 Why Self-Attention                    p.6   ref=0 | 'layer connects all positions with a constant number of sequentially executed operations, w'
   - attention §4 Why Self-Attention                    p.6   ref=0 | '4 Why Self-Attention In this section we compare various aspects of self-attention layers t'
