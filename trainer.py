import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    Gemma3ForCausalLM
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_from_disk, Dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import glob
import json
import shutil
import psutil
import signal
import sys
import argparse
import time
import numpy as np
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tempfile
import uuid
import gc

# ---------------- CONFIG ----------------
base_model_path = "C:/FineTune06/gemma-3-1b-it"
output_dir = "C:/FineTune06/gemma3-1b-Lora"
data_file = "C:/FineTune06/custom_data_gemma.jsonl"
temp_dataset_path = "C:/FineTune06/temp_dataset"
max_length = 512  # RTX 3060 için daha uzun sequence
batch_size = 2  # RTX 3060 için daha büyük batch
gradient_accumulation_steps = 8  # RTX 3060 için daha az accumulation
save_steps = 500
stop_training = False
use_ddp = False
max_epochs = 500
use_4bit = True  # 4-bit kuantizasyon kullan

# RTX 3060 12GB için özel bellek optimizasyonu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # CUDA 1 (RTX 3060)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_MEMORY_POOL'] = '1:536870912'  # 512MB bellek havuzu

# Hata ayıklama için (performansı düşürür, gerekirse açın)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ---------------- GPU MEMORY DETECTION ----------------
def detect_gpu_memory():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            memories = []
            for i in range(device_count):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                memories.append(total_memory)
            
            min_memory = min(memories) if memories else 0
            if min_memory <= 8 * 1024**3:  # 8GB
                return 8, 16
            elif min_memory <= 12 * 1024**3:  # 12GB
                return 12, 8  # RTX 3060 için daha az accumulation
            else:
                return 24, 4
        return 0, 16
    except Exception as e:
        print(f"GPU bellek tespiti sırasında hata: {e}")
        return 12, 8  # RTX 3060 varsayılan

gpu_memory_gb, default_gradient_steps = detect_gpu_memory()
print(f"En küçük GPU belleği: {gpu_memory_gb}GB, Önerilen gradient adımları: {default_gradient_steps}")

# ---------------- SIGNAL HANDLER ----------------
def signal_handler(sig, frame):
    global stop_training
    print("\nEğitim durduruluyor...")
    stop_training = True
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ---------------- DDP SETUP ----------------
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------- MEMORY USAGE ----------------
def print_memory_usage(rank=0):
    if rank == 0:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i} Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        ram = psutil.virtual_memory()
        print(f"RAM Kullanılan: {ram.used/1024**3:.2f} GB, Boş: {ram.available/1024**3:.2f} GB")

# ---------------- ANOMALY CLEAN ----------------
def clean_anomaly_files():
    anomaly_dir = "anomaly"
    if os.path.exists(anomaly_dir):
        for file in glob.glob(os.path.join(anomaly_dir, "*.pt")):
            try: 
                os.remove(file)
                print(f"Silinen anomaly dosyası: {file}")
            except Exception as e:
                print(f"Dosya silinemedi {file}: {e}")

# ---------------- EXTREME MEMORY CLEAN ----------------
def extreme_memory_clean():
    """RTX 3060 için agresif bellek temizleme"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # CUDA_VISIBLE_DEVICES='1' ayarından sonra fiziksel GPU1 cuda:0 olur
        with torch.cuda.device('cuda:0'):  # 'cuda:1' yerine 'cuda:0'
            torch.cuda.empty_cache()
            # RTX 3060 için daha yüksek bellek kullanımı
            torch.cuda.set_per_process_memory_fraction(0.7)  # %70 bellek kullanımı
            torch.cuda.empty_cache()

# ---------------- DATASET PREP ----------------
def prepare_dataset():
    processed = []
    with open(data_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        print(f"Toplam satır sayısı: {len(lines)}")
        
        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue
                
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                
                if 50 <= len(text) <= 500:  # RTX 3060 için daha uzun metinler
                    processed.append({"text": text})
                else:
                    print(f"Satır {line_num} atlandı: Uzunluk sınırı dışında ({len(text)} karakter)")
            except json.JSONDecodeError as e:
                print(f"JSON hatası (Satır {line_num}): {e}")
                continue
    
    print(f"İşlenen örnek sayısı: {len(processed)}")
    if len(processed) == 0:
        raise ValueError("Hiç geçerli veri işlenemedi!")
    
    dataset = Dataset.from_list(processed)
    dataset.save_to_disk(temp_dataset_path)
    return dataset

# ---------------- CHECKPOINT ----------------
def find_latest_checkpoint():
    if not os.path.exists(output_dir): 
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints: 
        return None
    
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    return os.path.join(output_dir, checkpoints[-1])

def save_checkpoint(model, tokenizer, optimizer, epoch, global_step, rank, learning_rate):
    if rank != 0:
        return
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}-step-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, "module") else model
    
    if isinstance(model_to_save, PeftModel):
        model_to_save.save_pretrained(checkpoint_dir)
    else:
        model_to_save.save_pretrained(checkpoint_dir)
    
    tokenizer.save_pretrained(checkpoint_dir)
    
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "learning_rate": learning_rate
    }
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f)
    
    print(f"Checkpoint kaydedildi: {checkpoint_dir}")

def load_checkpoint(model, checkpoint_dir, rank, device):
    if rank == 0:
        print(f"Checkpoint'ten yükleniyor: {checkpoint_dir}")
    
    epoch = 0
    global_step = 0
    learning_rate = 1e-4
    
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(training_state_path):
        try:
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            if "epoch" in training_state and "global_step" in training_state:
                epoch = training_state["epoch"]
                global_step = training_state["global_step"]
                if rank == 0:
                    print(f"Training state yüklendi: Epoch {epoch}, Step {global_step}")
            if "learning_rate" in training_state:
                learning_rate = training_state["learning_rate"]
                if rank == 0:
                    print(f"Learning rate yüklendi: {learning_rate}")
        except (json.JSONDecodeError, KeyError) as e:
            if rank == 0:
                print(f"Training state dosyası okunamadı: {e}")
    
    return epoch, global_step, learning_rate

# ---------------- CUSTOM DATA COLLATOR ----------------
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# ---------------- TRAIN FUNCTION ----------------
def train(rank, world_size, resume_from_checkpoint=None, args=None):
    global stop_training
    
    if use_ddp:
        setup_ddp(rank, world_size)
    
    # Belleği temizle
    extreme_memory_clean()
    
    # RTX 3060 için float16 kullanımı
    compute_dtype = torch.float16
    print(f"Using compute dtype {compute_dtype}")
    
    # CUDA_VISIBLE_DEVICES='1' ayarından sonra fiziksel GPU1 cuda:0 olur
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # "cuda:1" yerine "cuda:0"
    
    # Bellek optimizasyonu
    torch.backends.cudnn.benchmark = True  # RTX 3060 için benchmark aktif
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 aktif (RTX 3060 destekler)
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # RTX 3060 için daha yüksek bellek kullanımı
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)  # %70 bellek kullanımı
    
    print(f"Process {rank} başlatıldı. CUDA kullanılacak: {torch.cuda.is_available()}, Cihaz: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        local_files_only=True,
        max_seq_length=max_length
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit kuantizasyon yapılandırması - RTX 3060 için optimize
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_storage_dtype=torch.uint8,
        )
    else:
        bnb_config = None
    
    # Modeli yükle - RTX 3060 için optimize
    model = Gemma3ForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=compute_dtype,
        attn_implementation="eager",  # "flash_attention_2" yerine "eager"
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map={"": device},  # CUDA_VISIBLE_DEVICES='1' ayarından sonra device cuda:0 olur
        trust_remote_code=True,
        use_cache=False,
        local_files_only=True
    )
    
    # Modeli kuantizasyon için hazırla
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Gradient checkpointing aktifleştir
    model.gradient_checkpointing_enable()
    
    # Belleği temizle
    extreme_memory_clean()
    if torch.cuda.is_available():
        print(f"Model yüklendikten sonra GPU belleği: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # ---------------- LoRA ----------------
    # RTX 3060 için optimize edilmiş LoRA konfigürasyonu

    # ---------------- LoRA ----------------
    # RTX 3060 için optimize edilmiş LoRA konfigürasyonu
    peft_config = LoraConfig(
        r=128,  # Yüksek rank
        lora_alpha=256,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "input_layernorm", "post_attention_layernorm"  # Ek LayerNorm
        ],
        lora_dropout=0.002,  # Yüksek dropout lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Checkpoint'ten devam etme
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        try:
            print(f"LoRA adaptörü yükleniyor: {resume_from_checkpoint}")
            base_model = Gemma3ForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=compute_dtype,
                attn_implementation="eager",  # "flash_attention_2" yerine "eager"
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map={"": device},
                trust_remote_code=True,
                use_cache=False,
                local_files_only=True
            )
            model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
            print("LoRA adaptörü başarıyla yüklendi.")
        except Exception as e:
            print(f"Checkpoint yüklenemedi: {e}")
            print("Yeni LoRA adaptörü oluşturuluyor...")
            model = get_peft_model(model, peft_config)
    else:
        print("Yeni LoRA adaptörü oluşturuluyor...")
        model = get_peft_model(model, peft_config)
    
    # Eğitilebilir parametreleri yazdır
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / all_param
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {percentage:.4f}")
    
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, static_graph=True)
    
    # Dataset
    if rank == 0:
        valid_dataset = False
        if os.path.exists(temp_dataset_path):
            try:
                test_load = load_from_disk(temp_dataset_path)
                valid_dataset = True
                print("Mevcut dataset doğrulandı, kullanılacak.")
            except Exception as e:
                print(f"Mevcut dataset geçersiz: {e}")
                print("Yeni dataset oluşturulacak...")
        
        if not valid_dataset:
            print("Yeni dataset hazırlanıyor...")
            if os.path.exists(temp_dataset_path):
                shutil.rmtree(temp_dataset_path)
            dataset = prepare_dataset()
        else:
            dataset = load_from_disk(temp_dataset_path)
        
        clean_anomaly_files()
        
        def tokenize_fn(examples):
            texts = examples["text"]
            
            try:
                tok = tokenizer(
                    texts, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                tok["labels"] = tok["input_ids"].clone()
                return tok
            except Exception as e:
                print(f"Tokenizer hatası: {e}")
                return {
                    "input_ids": torch.zeros((len(texts), max_length), dtype=torch.long),
                    "attention_mask": torch.zeros((len(texts), max_length), dtype=torch.long),
                    "labels": torch.zeros((len(texts), max_length), dtype=torch.long)
                }
        
        tokenized = dataset.map(
            tokenize_fn, 
            batched=True, 
            batch_size=8,  # RTX 3060 için daha büyük batch
            remove_columns=dataset.column_names,
            load_from_cache_file=False
        )
        
        if use_ddp:
            dist.barrier()
    else:
        if use_ddp:
            dist.barrier()
        
        if os.path.exists(temp_dataset_path):
            try:
                tokenized = load_from_disk(temp_dataset_path)
            except Exception as e:
                print(f"Tokenize edilmiş veri seti yüklenemedi: {e}")
                raise FileNotFoundError("Tokenize edilmiş veri seti bulunamadı")
        else:
            raise FileNotFoundError("Temp dataset dizini bulunamadı")

    print(f"Process {rank}: Veri seti yüklendi, {len(tokenized)} örnek var.")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    if use_ddp:
        sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    
    dataloader = DataLoader(
        tokenized, 
        batch_size=batch_size, 
        sampler=sampler, 
        collate_fn=data_collator, 
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Optimizer
    initial_lr = args.learning_rate if args and hasattr(args, 'learning_rate') else 2e-5  # RTX 3060 için daha yüksek LR
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # Öğrenme oranı zamanlayıcısı
    from transformers import get_linear_schedule_with_warmup
    
    steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    num_training_steps = steps_per_epoch * max_epochs
    num_warmup_steps = int(0.03 * num_training_steps)
    
    start_epoch = 0
    global_step = 0
    current_lr = initial_lr
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        try:
            start_epoch, global_step, loaded_lr = load_checkpoint(model, resume_from_checkpoint, rank, device)
            start_epoch += 1
            
            if args and hasattr(args, 'learning_rate'):
                target_lr = args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
                current_lr = target_lr
                if rank == 0:
                    print(f"🔄 Learning rate sıfırlandı: {loaded_lr} -> {target_lr}")
            else:
                current_lr = loaded_lr
                if rank == 0:
                    print(f"📈 Learning rate yüklendi: {current_lr}")
            
            if rank == 0:
                print(f"🚀 Checkpoint'ten devam ediliyor: Epoch {start_epoch}, Step {global_step}")
        except Exception as e:
            print(f"❌ Checkpoint yüklenirken hata: {e}")
            print("🔄 Yeniden başlatılıyor...")
            start_epoch = 0
            global_step = 0
            current_lr = initial_lr
    
    # ÖNEMLİ DÜZELTME: Checkpoint'ten devam ediyorsak eğitim adımlarını artır
    if global_step > 0:
        # Kalan eğitim adımlarını hesapla
        remaining_epochs = max_epochs - start_epoch
        num_training_steps = global_step + (steps_per_epoch * remaining_epochs)
        if rank == 0:
            print(f"🔄 Toplam eğitim adımı güncellendi: {num_training_steps} (mevcut step: {global_step})")
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Checkpoint'ten devam ediyorsak scheduler'ı güncelle
    if global_step > 0:
        lr_scheduler.last_epoch = global_step
        if rank == 0:
            print(f"🔄 Learning rate scheduler güncellendi: mevcut step {global_step}")
    
    scaler = GradScaler()
    
    best_loss = float('inf')
    patience = 15  # RTX 3060 için daha az patience
    patience_counter = 0
    
    if rank == 0:
        print("🎯 Eğitim başlatılıyor...")
        print(f"📊 Toplam veri noktası: {len(tokenized)}")
        print(f"📈 Epoch başına adım sayısı: {steps_per_epoch}")
        print(f"🎯 Toplam eğitim adımı: {num_training_steps}")
        print(f"🔥 Warmup adımı: {num_warmup_steps}")
        print(f"📚 Başlangıç öğrenme oranı: {current_lr}")
        print(f"🏁 Başlangıç epoch'u: {start_epoch}")
        print(f"🎯 Maksimum epoch: {max_epochs}")
        print(f"🔢 Max sequence length: {max_length}")
        print(f"🎯 Compute dtype: {compute_dtype}")
        print(f"🔢 4-bit kuantizasyon: {use_4bit}")
        print(f"💾 Checkpoint kaydetme sıklığı: Her {save_steps} adımda bir")
        print_memory_usage()
    
    model.train()
    
    try:
        for epoch in range(start_epoch, max_epochs):
            if stop_training:
                print("⏹️ Eğitim kullanıcı tarafından durduruldu.")
                break
                
            if use_ddp:
                sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"📅 Epoch {epoch+1}/{max_epochs}")
                epoch_start_time = time.time()
            
            total_loss = 0.0
            steps_processed = 0
            
            for step, batch in enumerate(dataloader):
                if stop_training:
                    print("⏹️ Eğitim kullanıcı tarafından durduruldu.")
                    break
                
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Bellek temizleme - her 10 adımda bir
                if rank == 0 and step % 10 == 0:
                    extreme_memory_clean()
                
                with autocast(device_type="cuda", dtype=compute_dtype):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    total_loss += loss.item() * gradient_accumulation_steps
                    steps_processed += 1
                    
                    if rank == 0:
                        current_loss = loss.item() * gradient_accumulation_steps
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f"📊 Step {global_step}/{num_training_steps}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")
                    
                    if rank == 0 and global_step % save_steps == 0:
                        save_checkpoint(model, tokenizer, optimizer, epoch, global_step, rank, current_lr)
                        print(f"🎯 Step {global_step} checkpoint kaydedildi!")
            
            if steps_processed > 0:
                avg_loss = total_loss / steps_processed
            else:
                avg_loss = float('inf')
                print("⚠️ Uyarı: Bu epoch'ta hiç adım işlenmedi!")
            
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                print(f"✅ Epoch {epoch+1}/{max_epochs} tamamlandı, Ortalama Kayıp: {avg_loss:.4f}, Süre: {epoch_time:.2f}s, LR: {current_lr:.2e}")
                print_memory_usage()
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    save_checkpoint(model, tokenizer, optimizer, epoch, global_step, rank, current_lr)
                    print(f"🏆 En iyi model güncellendi! Kayıp: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"⏳ Early stopping sayacı: {patience_counter}/{patience}")
                    
                    if patience_counter >= patience:
                        print(f"🛑 Early stopping tetiklendi! {patience} epoch boyunca iyileşme olmadı.")
                        stop_training = True
                
                extreme_memory_clean()
            
            if stop_training:
                break
                
    except Exception as e:
        print(f"❌ Eğitim sırasında hata: {e}")
        if rank == 0:
            save_checkpoint(model, tokenizer, optimizer, epoch, global_step, rank, current_lr)
        raise
    
    if use_ddp:
        cleanup_ddp()
    
    if rank == 0:
        print("\n" + "="*50)
        print("🎉 EĞİTİM TAMAMLANDI!")
        print(f"📊 Orijinal veri seti: {len(tokenized)} örnek")
        print(f"📈 Son öğrenme oranı: {current_lr:.2e}")
        print(f"💾 Model başarıyla kaydedildi: {output_dir}")
        print(f"🔢 Max sequence length: {max_length}")
        print(f"🎯 Compute dtype: {compute_dtype}")
        print(f"🔢 4-bit kuantizasyon: {use_4bit}")
        print("="*50)

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma 3 model')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Checkpoint yolunu belirtin')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Tokenizasyon için maksimum uzunluk (varsayılan: 512)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (varsayılan: 2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient biriktirme adımları (varsayılan: 8)')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Checkpoint kaydetme sıklığı (varsayılan: 500)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Başlangıç öğrenme oranı (varsayılan: 2e-5)')
    parser.add_argument('--max_epochs', type=int, default=500,
                       help='Maksimum epoch sayısı (varsayılan: 500)')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                       help='4-bit kuantizasyon kullan (varsayılan: True)')
    parser.add_argument('--enable_ddp', action='store_true',
                       help='DDP kullanımını aktif et (çoklu GPU için)')
    parser.add_argument('--force_restart', action='store_true',
                       help='Yeniden başlat - checkpoint kullanma')
    args = parser.parse_args()
    
    # Global değişkenleri güncelle
    global max_length, batch_size, gradient_accumulation_steps, save_steps, use_ddp, max_epochs, use_4bit
    max_length = args.max_length
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    save_steps = args.save_steps
    max_epochs = args.max_epochs
    use_ddp = args.enable_ddp
    use_4bit = args.use_4bit
    
    world_size = torch.cuda.device_count() if use_ddp else 1
    if world_size == 0:
        world_size = 1
        use_ddp = False
    
    print(f"🖥️ Kullanılabilir GPU sayısı: {world_size}")
    print(f"🔗 DDP kullanılacak: {use_ddp}")
    print(f"🔢 4-bit kuantizasyon kullanılacak: {use_4bit}")
    print(f"📈 Kullanılacak gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"📚 Kullanılacak başlangıç öğrenme oranı: {args.learning_rate}")
    print(f"🎯 Kullanılacak maksimum epoch sayısı: {max_epochs}")
    print(f"🔢 Kullanılacak max sequence length: {max_length}")
    print(f"💾 Checkpoint kaydetme sıklığı: Her {save_steps} adımda bir")
    
    if args.force_restart:
        args.resume_from_checkpoint = None
        print("🔄 Zorla yeniden başlatılıyor - checkpoint kullanılmayacak")
    elif args.resume_from_checkpoint is None:
        args.resume_from_checkpoint = find_latest_checkpoint()
    
    if args.resume_from_checkpoint:
        print(f"🚀 Kaldığı yerden devam ediliyor: {args.resume_from_checkpoint}")
    else:
        print("🆕 Yeni eğitim başlatılıyor")
    
    if use_ddp and world_size > 1:
        print(f"🔗 DDP ile {world_size} GPU kullanılacak")
        try:
            mp.spawn(train, args=(world_size, args.resume_from_checkpoint, args), nprocs=world_size, join=True)
        except KeyboardInterrupt:
            print("⚠️ Ana işlem KeyboardInterrupt ile sonlandırıldı.")
    else:
        print("💻 Tek GPU veya CPU ile çalışılacak")
        try:
            train(0, 1, args.resume_from_checkpoint, args)
        except KeyboardInterrupt:
            print("⚠️ Ana işlem KeyboardInterrupt ile sonlandırıldı.")

if __name__=="__main__":
    main()
