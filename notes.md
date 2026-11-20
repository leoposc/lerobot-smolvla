# SmolVLA implementation details


#### RoPE Self-attention

RoPE is applied to the concatenated QK from both AE and VLM. Is this weird?

The position ids increase incrementally by 1 (except there is padding). The sequence length differs if the VLM does not have a corresponding AE layer. Does anything weird happen if the sequence length is different? The position_ids should stay the same, since the AE tokens are appended to the others. But if a camera is missing, we add padding, therefore all following tokens receive a different position_id. 

