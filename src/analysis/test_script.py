import torch

# Bu komut 'True' yazdırmalıdır.
print(f"CUDA (GPU) desteği var mı? : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Eğer 'True' ise, hangi GPU'yu gördüğünü yazdıralım
    print(f"Kullanılan GPU Adı: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch, CUDA destekli bir GPU bulamadı. Yalnızca CPU kullanılacak.")