> Bu belge nv-tlabs/ardy deposundaki [`MotionCorrection/README.md`](https://github.com/nv-tlabs/ardy/blob/main/MotionCorrection/README.md) dosyasının Türkçe çevirisidir (Apache-2.0, commit `e706e27`).

# motion_correction: hareket düzeltme kütüphanesi

Bağımsız `correct_motion` uygulaması, Python bağlamaları (bindings) olan küçük bir C++ hareket yörüngesi düzeltme kütüphanesi olarak paketlenmiştir.

ARDY'nin ana kurulumu (`pip install -e ".[all]"`) bu eklentiyi otomatik derler. Demodaki **Enable Post-Processing** seçeneği ve `scripts/generate.py` betiğinin hareket son işlemesi, ayak kaymasını (foot skating) azaltmak ve kısıtları tutturmak için bu düzeltmeyi kullanır. Ayrıntı için [Türkçe README](../README.md#-kurulum) sayfasına bakın.

## Kurulum rehberi

### Ön koşullar

C++17 uyumlu bir derleyiciniz (GCC 7.0+, Clang 5.0+ ya da MSVC 2017+) ve CMake 3.15+ kurulu olsun. Windows'ta MinGW-w64 ya da C++ araçlarıyla birlikte Visual Studio kurun. Linux'ta `build-essential` ve `cmake` paketlerini kurun.

Bu proje ek üçüncü taraf açık kaynak yazılım projelerini indirir ve kurar. Kullanmadan önce bu projelerin lisans koşullarını inceleyin.

### Derleme ve kurulum

#### Standart kurulum
```bash
pip install .
```

#### Geliştirme kurulumu
```bash
pip install -e .
```

### Kurulumu doğrulayın

```python
import motion_correction
print("Installation successful!")
```
Basit bir test için `python run_test.py` komutunu da çalıştırabilirsiniz.

---

**Unitree G1 Türkiye:** [Unitree G1 insansı robot ürün sayfası](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1) · [G1 EDU için teklif alın](https://www.robotlar.org/tr/fiyat/unitree-g1-edu-fiyat) · Unitree Robotics Türkiye Resmi Distribütörü Robotlar.org
