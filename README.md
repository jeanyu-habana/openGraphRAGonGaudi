# openGraphRAGonGaudi
Examples for RAG Construction and Queries Locally on Intel Gaudi. The examples are compatible with Llama-Index and Optimum-Habana, LlamaIndex, and NebulaGraph Database. The examples have been tested on a single Gaudi2 device.

# Installation Instructions

To set up your environment, please install the following libraries by running the provided commands.

## Libraries
1. **Optimum with Habana support**
   - Install Optimum with Habana support:
     ```bash
     pip install --upgrade-strategy eager optimum[habana]
     ```

2. **Install from `requirements.txt`**
     ```bash
     pip install -r requirements.txt
     ```
3. **Setup and Configure NebulaGraph Database**
   - Install NebulaGraph database with instructions [here](https://docs.nebula-graph.io/3.0.0/2.quick-start/2.install-nebula-graph/)
   - Run the Nebula Console
     ```bash
     $ ~/.nebula-up/console.sh
     ```
   - Create and configure a nebula space (e.g., llamaindex) from the nebula console:
     ```bash
     CREATE SPACE llamaindex(vid_type=FIXED_STRING(1024), partition_num=1, replica_factor=1); 
     USE llamaindex;  
     CREATE TAG entity(name string); 
     CREATE EDGE relationship(relationship string); 
     CREATE TAG INDEX entity_index ON entity(name(256)); 
     ```
4. **Update Nebula Connection Information in gaudi_graph_constructor.py**
   - NEBULA_USER
   - NEBULA_PASSWORD
   - NEBULA_ADDRESS
   - Nebula space

## Example
```bash
# Single-card model inference
python gaudi_graph_constructor \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--max_new_tokens 100 \
--bf16 \
--use_hpu_graphs \
--use_kv_cache
```
## Sample Output
/22/2024 14:03:21 - INFO - __main__ - Single-device run. 
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.02s/it] 
============================= HABANA PT BRIDGE CONFIGURATION =========================== 
PT_HPU_LAZY_MODE = 1 
PT_RECIPE_CACHE_PATH = 
PT_CACHE_FOLDER_DELETE = 0 
PT_HPU_RECIPE_CACHE_CONFIG = 
PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807 
PT_HPU_LAZY_ACC_PAR_MODE = 1 
PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0 
---------------------------: System Configuration :--------------------------- 
Num CPU Cores : 10 
CPU RAM       : 100936728 KB 
------------------------------------------------------------------------------ 
You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers 
08/22/2024 14:03:33 - INFO - __main__ - Args: Namespace(device='hpu', model_name_or_path='/home/ubuntu/jean/models/mistral', bf16=True, max_new_tokens=100, max_input_tokens=0, batch_size=1, warmup=3, n_iterations=5, local_rank=0, use_kv_cache=True, use_hpu_graphs=True, dataset_name=None, column_name=None, do_sample=False, num_beams=1, trim_logits=False, seed=27, profiling_warmup_steps=0, profiling_steps=0, profiling_record_shapes=False, prompt=None, bad_words=None, force_words=None, assistant_model=None, peft_model=None, num_return_sequences=1, token=None, model_revision='main', attn_softmax_bf16=False, output_dir=None, bucket_size=-1, bucket_internal=False, dataset_max_samples=-1, limit_hpu_graphs=False, reuse_cache=False, verbose_workers=False, simulate_dyn_prompt=None, reduce_recompile=False, use_flash_attention=False, flash_attention_recompute=False, flash_attention_causal_mask=False, flash_attention_fast_softmax=False, book_source=False, torch_compile=False, ignore_eos=True, temperature=1.0, top_p=1.0, const_serialization_path=None, disk_offload=False, trust_remote_code=False, quant_config='', world_size=0, global_rank=0) 
08/22/2024 14:03:33 - INFO - __main__ - device: hpu, n_hpu: 0, bf16: True 
08/22/2024 14:03:33 - INFO - __main__ - Model initialization took 12.630s 
08/22/2024 14:03:33 - INFO - sentence_transformers.SentenceTransformer - Load pretrained SentenceTransformer: thenlper/gte-large 
/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`. 
  warnings.warn( 
08/22/2024 14:03:36 - INFO - sentence_transformers.SentenceTransformer - Load pretrained SentenceTransformer: thenlper/gte-large 
  
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: ad445256-f355-4356-aaef-762581b7c970: In a mid-credits scene, the new Guardians—Rocket, Groot, Kraglin, Cosmo, Adam... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: c4a20d41-52c4-439a-acc5-0bc99a4ee2ca: By then, Bautista said he had not read a script for Vol. 3 and was unsure if ... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: 502ad007-b1f0-4df8-99bd-f2e80167718f: === Critical response === 
The review aggregator website Rotten Tomatoes repor... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: e153f263-0245-4c5a-bce2-ccfec90ebfb2: The Guardians travel to Orgocorp's headquarters to find the switch's override... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: b931d5bd-204c-4f4c-8f52-75352225155b: Kraglin fires on Arête with Knowhere and then helps to save Knowhere's citize... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: f7019e68-fec7-4f8e-8312-0c5f046d2ad6: Cameo appearances in the film include Lloyd Kaufman as Gridlemop, a Krylorian... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: c61d4ece-7da5-4404-9910-8bf7659d5521: === Home media === 
Guardians of the Galaxy Vol. 3 was released by Walt Disney... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: 94795502-9146-4445-9e5e-2ca98d40fdf3: Hermanns also said the second trailer "goes a step further in highlighting th... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: c3a4acfd-35e4-4619-8332-09307cfb2c63: Guardians of the Galaxy Vol. 3 (marketed as Guardians of the Galaxy Volume 3)... 
08/22/2024 14:09:22 - INFO - llama_index.core.indices.knowledge_graph.retrievers - > Querying with idx: ab4a35f7-661b-4b36-8f75-6007e98becaa: == Notes == 
== References == 
== External links == 
Official website  at M... 
Chris Pratt, Zoe Saldaña, Dave Bautista, Karen Gillan, Pom Klementieff, Vin Diesel, Bradley Cooper, Will Poulter, Sean Gunn, Chukwudi Iwuji, Linda Cardellini, Nathan Fillion, and Sylvester Stallone. 


