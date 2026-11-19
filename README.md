<div align="center">

# NVIDIA ARDY Türkçe: Unitree G1 ve İnsansı Karakterler için Gerçek Zamanlı Etkileşimli Hareket Üretimi

**ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation**

**Otoregresif difüzyon · Akışlı metinden harekete · Kinematik kısıtlar · Core, Unitree G1 ve SOMA modelleri**

**Unitree Robotics Türkiye Resmi Distribütörü Robotlar.org · [www.robotlar.org](https://www.robotlar.org)**

👉 **Unitree G1 ürün sayfası: [robotlar.org/tr/insansi-robotlar/unitree-g1](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1) · G1 EDU için teklif: [robotlar.org/tr/fiyat/unitree-g1-edu-fiyat](https://www.robotlar.org/tr/fiyat/unitree-g1-edu-fiyat)**

📖 **Türkçe rehber: [NVIDIA ARDY gerçek zamanlı hareket rehberi](https://www.robotlar.org/tr/rehber/nvidia-ardy-gercek-zamanli-hareket-rehberi)**

[Kurulum](#-kurulum) · [Kontrol Noktaları](#-kontrol-noktaları-checkpoints) · [Etkileşimli Demo](#-etkileşimli-demo) · [Komut Satırı](#-komut-satırından-üretim) · [Kıyaslama](#-hareket-üretimi-kıyaslaması-benchmark) · [Belgeler](#-belgeler) · [G1 ile Başlangıç](#-unitree-g1-ile-ardy-türkiyede-nereden-başlanır) · [Unitree Ürünleri](#-unitree-ürün-ailesi-türkiye)

</div>

<p align="center">
  <a href="https://research.nvidia.com/labs/sil/projects/ardy/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Proje sayfası"></a>
  <a href="https://github.com/nv-tlabs/ardy/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="Lisans"></a>
  <img src="https://raw.githubusercontent.com/nv-tlabs/ardy/main/assets/banner.png" alt="NVIDIA ARDY ile metin istemleri ve kinematik kısıtlarla gerçek zamanlı insansı hareket üretimi" width="100%">
</p>

> Bu depo, NVIDIA'nın resmi [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) projesinin belgelerinin Türkçe çevirisidir. Çeviri, kaynak deponun `e706e27` commit'ine (15 Ekim 2026) dayanır. Kaynak kod, kontrol noktaları, iskelet varlıkları ve görseller orijinal depoda kalır. Burada yalnız Türkçe anlatım bulunur ve görseller orijinal depodan gösterilir.
>
> **Lisans dağılımı:** kod Apache-2.0 lisanslıdır. Model kontrol noktaları ve veriler, Hugging Face indirme sayfalarında belirtilen ayrı lisanslarla gelir (modeller için NVIDIA Open Model Agreement). Bu çeviri Apache-2.0 koşullarıyla, değişiklik bildirimiyle yayımlanır. Ayrıntı [Lisans](#-lisans) bölümünde.
>
> **Kodu çalıştırmak için orijinal depoyu klonlayın:** [github.com/nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)
>
> ARDY, NVIDIA'nın projesidir. Çeviri ve Türkçe teknik destek: Unitree Türkiye resmi distribütörü [Robotlar.org](https://www.robotlar.org).

ARDY, etkileşimli hareket üretimi için tasarlanmış otoregresif bir difüzyon modelidir (autoregressive diffusion model). Çevrim içi metin istemlerini (online text prompting) ve esnek, uzun ufuklu kinematik kısıtları gerçek zamanlı tepkiyle destekler. Kısıtlar kök yollarını ve ara noktaları (root paths/waypoints), tüm gövde anahtar karelerini (full-body keyframes) ve seyrek eklem konum ve dönüşlerini kapsar.

Orijinal depo, [SIGGRAPH makalesinde](https://research.nvidia.com/labs/sil/projects/ardy/) tanıtılan önceden eğitilmiş ARDY modelleriyle çalışmak için kodu, kontrol noktalarını ve demoları sunar.

## 📥 Kurulum

> Depo ağırlıklı olarak Ubuntu Linux 22.04, RTX 4090, nvidia-driver-575 ve Python 3.11 ile test edildi.

Temiz bir Python 3.10+ ortamında (conda ya da venv) önce makinenizin CUDA sürümüne uyan bir **PyTorch** (>= 2.4) derlemesi kurun. Ardından ARDY'yi tüm isteğe bağlı özellikleriyle düzenlenebilir kipte (editable mode) kurun:

```bash
conda create -n ardy python=3.11 -y
conda activate ardy
# Önce CUDA sürümünüze uygun PyTorch'u kurun, bkz. https://pytorch.org/get-started/locally/
# Örneğin:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[all]"
```

<details>
<summary>Kurulum ayrıntıları: neden önce PyTorch, derleme gereksinimleri ve kısmi kurulumlar</summary>

- **Neden önce PyTorch:** ARDY yalnız `torch>=2.4` ister. PyTorch'u önce kendiniz kurduğunuzda GPU'nuza ve sürücünüze uyan CUDA derlemesini seçersiniz. ARDY kurulumu da varsayılan bir wheel çekmek yerine elinizdeki uyumlu PyTorch'u korur.
- **Derleme gereksinimleri:** kurulum, birlikte gelen hareket düzeltme (motion-correction) C++ eklentisini derler. Bunun için CMake >= 3.15 ve bir C++17 derleyicisi gerekir (Ubuntu'da `sudo apt install cmake build-essential`). Eklentinin kendi kurulum notları [Hareket düzeltme kütüphanesi](docs/HAREKET-DUZELTME.md) belgesinde.
- **Kısmi kurulumlar:** isteğe bağlı bağımlılık grupları tek tek de kurulabilir:
  - `pip install -e .`: yalnız çekirdek model çıkarımı
  - `pip install -e ".[demo]"`: etkileşimli demoyu ekler (viser, gradio)
  - `pip install -e ".[trt]"`: TensorRT hızlandırmasını ekler
- **TensorRT gereksinimleri:** `[trt]` eki, NVIDIA sürücüsü >= 525 (CUDA 12 destekli; CUDA çalışma zamanının kendisi pip ile birlikte gelir) ve kurulum sırasında `pypi.nvidia.com` erişimi ister. Bu koşulları karşılamayan sistemlerde bunun yerine `.[demo]` kurun ve demoda TensorRT dışı bir hızlandırma kipi seçin.

</details>

### Metin kodlayıcı için Hugging Face jetonunu ayarlayın

Metin kodlayıcı (text encoder), erişimi onaya bağlı [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) modelini kullanır. Bunun için iki koşul gerekir:

- Hugging Face hesabınıza model sayfasına erişim izni verilmiş olmalı.
- Çalışma sırasında bir Hugging Face jetonu (token) sağlamalısınız.

Llama deposuna erişim aldıktan sonra [Hugging Face jeton ayarlarından](https://huggingface.co/settings/tokens) bir erişim jetonu oluşturun. Ardından ya komut satırından oturum açın:

```bash
hf auth login
```

ya da jetonu `~/.cache/huggingface/token` dosyasına yapıştırın. `hf` kurulu değilse önce onu kurun:

```bash
pip install --upgrade huggingface_hub
```

### 💾 Kontrol Noktaları (Checkpoints)

ARDY kontrol noktaları, FPS değeri ve tahmin ufku (prediction horizon) uzunluğu farklı çeşitli iskeletler üzerinde eğitilmiş olarak sunulur.

> Not: modeller demoda kullanıldıklarında otomatik indirilir, bu yüzden onları elle indirmeniz gerekmez.

| Model | İskelet | Eğitim Verisi | FPS | Ufuk | Yayın Tarihi | Hugging Face | Lisans |
|:-------|:-------------|:------:|:------:|:------:|:------:|:-------------:|:-------------:|
| **ARDY-Core-RP-20FPS-Horizon40** | Core | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 20 | 40 | 10 Temmuz 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-Core-RP-20FPS-Horizon8** | Core | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 20 | 8 | 10 Temmuz 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon8) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-G1-RP-25FPS-Horizon52** | Unitree G1 | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 25 | 52 | 10 Temmuz 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-G1-RP-25FPS-Horizon8** | Unitree G1 | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 25 | 8 | 10 Temmuz 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon8) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-SOMA-RP-30FPS-Horizon60** | [SOMA](https://github.com/NVlabs/SOMA-X) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 30 | 60 | 15 Ekim 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-Horizon60) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |
| **ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60** | [SOMA](https://github.com/NVlabs/SOMA-X) | [Bones Rigplay 1](https://bones.studio/datasets#rp01) | 30 | 60 | 15 Ekim 2026 | [Bağlantı](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60) | [NVIDIA Open Model](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) |

> *KimodoRep* çeşidi, ham kalça (raw-hips) tabanlı ARDY gösterimi yerine Kimodo hareket gösterimiyle (Kimodo modellerinin kullandığı, kök özniteliklerini yumuşatılmış bir kök yörüngesine göre veren gösterim) eğitildi. Bu çeşit, `--motion_rep kimodo` ile kurulan test paketleriyle eşleşir.

### [İsteğe bağlı] Veri seti

Veri indirmek yalnız kinematik kısıtlı üretim demolarını çalıştırmak için gerekir.

**Bones SEED hareket verisi:**
Kinematik kısıtlı üretim demosu, kısıtları [Bones SEED](https://huggingface.co/datasets/bones-studio/seed) veri setindeki hareket dizilerinden örnekler. Hareket verisi G1 için CSV biçiminde sunulur. Örnekleme sırasında karşılık gelen metin açıklamaları meta veri CSV dosyasından alınır.

Bones SEED veri setini indirin ve deponun kök dizinindeki `datasets/bones-seed/` klasörüne koyun. Klasör yapısı şöyle olmalı:
```
datasets/bones-seed/
  g1/csv/
  metadata/
    seed_metadata_v004.csv
```

---

## 🎮 Etkileşimli Demo

Orijinal depo, akışlı metin istemleri ile etkileşimli uzamsal kısıtları birleştirerek insansı karakteri gerçek zamanlı yönetmenizi sağlayan etkileşimli bir demo sunar. Demonun özellikleri:

- Web tarayıcısı görüntü alanında (viewport) izlenen çevrim içi hareket üretimi
- Akışlı metinden harekete (streaming text-to-motion)
- Kinematik kısıtlı hareket üretimi (tüm gövde anahtar kareleri, kök yörüngesi ve ara noktalar, uç işlevci (end-effector) eklemleri)
- Fare ve klavyeyle yürüyüş/hareket (lokomosyon) kontrolü

### Etkileşimli demoyu başlatın

```bash
python scripts/run_demo.py
```

Demoyu sık sık yeniden başlatacaksanız metin kodlayıcı API hizmetini arka planda çalıştırmak işinizi kolaylaştırır. Böylece demo her açılışta metin kodlayıcıyı yeniden kurmaz. Ayrı bir terminalde şunu çalıştırın:
```bash
python scripts/run_text_encoder_server.py
```

### Hızlı başlangıç
- **Arayüzü açın:** tarayıcınızda `http://localhost:2333` adresine gidin. Kamerayı fareyle sol sürükleyerek döndürür, sağ sürükleyerek kaydırır, tekerlekle yakınlaştırırsınız.
- **Bir model yükleyin:** **Model** sekmesindeki **Model Directory** açılır listesinden bir kontrol noktası seçin ve **Load Model** düğmesine tıklayın. TensorRT derlemesi açıksa ilk yükleme birkaç dakika sürebilir. Elle bir kontrol noktası seçmezseniz varsayılan kontrol noktası yüklenir.
- **Oynatmayı başlatın:** hareketi oynatmak ya da durdurmak için `Space` tuşuna basın ya da **Play** düğmesine tıklayın.
- **Metin istemleri:** **Prompt List** içindeki hazır bir isteme tıklayın ya da **Prompt Text** alanına kendi isteminizi yazıp **Update Text Prompt** düğmesine tıklayın.
- **Fareyle yürüyüş kontrolü:** ara nokta kipini açmak için `p` tuşuna basın (ya da **Generate** sekmesindeki **Waypoint** klasöründe **Enable Waypoint Mode** kutusunu işaretleyin). Ardından yürüyüşü yönlendirmek için görüntü alanına tıklayarak ara noktalar yerleştirin.
- **Klavyeyle hız kontrolü:** hedef hız kontrolünü açmak için `t` tuşuna basın. Yön ok tuşlarıyla verilir: sol ve sağ yönü döndürür, yukarı ve aşağı hızı artırır ya da azaltır.
- **Kinematik kısıtlar:** **Motion file path** alanında (**Generate** sekmesindeki **Constraints** klasöründe) belirtilen veri seti dizisinden kinematik kısıt örneklemek için `z` tuşuna basın. Hangi kategorilerin örnekleneceğini kısıt türü onay kutularıyla seçersiniz.
- **Oturumları dışa aktarın ve yükleyin:** kısıtları ve hareketi temel düzeyde dışa aktarmak ve yüklemek için **IO** sekmesini kullanın.

Etkileşimli demo hakkındaki ayrıntılar [Arayüz başvurusu](#arayüz-başvurusu-gui-reference) bölümünde.

### Arayüz başvurusu (GUI Reference)

<details>
<summary>Ayrıntılı arayüz kullanımı için tıklayıp açın</summary>

#### Kısayol tuşları

| Tuş | Eylem |
|-----|--------|
| Space | Oynat ya da duraklat |
| j | Önceki kare |
| k | Sonraki kare |
| r | Otomatik kamerayı geçerli kareyi izleyecek biçimde sıfırla |
| t | Hedef hız kontrolünü ve ok tuşu katmanını aç ya da kapat |
| p | Ara nokta kontrol kipini aç ya da kapat |
| z | Belirtilen hareket dosyasından kinematik kısıt örnekle |
| Ok tuşları | Hedef hızın yönünü ve büyüklüğünü ayarla (hedef hız kontrolü açıkken) |
| h | Klavye kısayolu katmanını göster ya da gizle |

#### Sekme: Playback

- **Native FPS:** yüklenen modelin hareket FPS değeri.
- **Enable Auto-Replan:** kapalıyken hareket otomatik üretilmez ve son üretilen karede durur.

#### Sekme: Text

- **Prompt List:** örnek istemleri içeren hazır istem düğmeleri. Bir düğmeye tıklamak etkin metin istemini ve gömmesini (embedding) doğrudan günceller. Metin istemlerini çalışma sırasında seçebilir, **Prompt Text** alanından kendi isteminizi de girebilirsiniz.
- **Prompt Text:** metin istemini buraya yazın. İstem, Update düğmesine tıklayana kadar etkinleşmez.
- **Update Text Prompt:** etkin metin istemini ve gömmeyi güncellemek için tıklayın.

#### Sekme: Generate

- **Restart:** tüm hareketi temizler ve üretimi ilk kareden yeniden başlatır.
- **Restart From Now:** geçerli kareden sonraki hareketi temizler ve üretimi geçerli kareden sürdürür.
- **Clear All Constraints:** tüm kinematik kısıtları temizler.
- **Initial Body Transform:** başlangıçtaki kök konumunu ve yönelimini ayarlayan bir gizmo.
- **Constraints**
  - Motion file path: `datasets/bones-seed/` içinden bir CSV (G1) hareket dosyasının yolu. Kinematik kısıtlar bu hareket dosyasından örneklenir.
  - Random Motion File düğmesi: `datasets/bones-seed/` içinden yüklü iskelete uyan rastgele bir hareket dosyası seçer ve Motion file path alanını doldurur.
  - Crop to 10s: açıksa, yüklenen hareket 10 saniyeden uzunsa rastgele 10 saniyelik bir parçaya kırpılır.
  - Kısıt türü onay kutuları [full body, hands, feet, hands+feet, 2D waypoints, 2D trajectory]: açık olan kategori için kısıt örneklenir.
  - Max keyframe number: örneklemede izin verilen en fazla anahtar kare kısıtı sayısı
  - Continue from current frame: açıksa yeni kısıtlar yeniden başlatmadan örneklenir. Kapalıysa üretimden önce hareket geçmişi temizlenir.
  - Sample Constraints düğmesi (kısayol: `z`): basıldığında kinematik kısıtlar belirtilen hareket dosyasından örneklenir. Kinematik kısıt örneklemesi, Bones Seed veri setinde bulunan G1 iskeletiyle çalışır.
- **Waypoint** (fareyle yürüyüş kontrolü)
  - Waypoint mode (kısayol: `p`): fare tıklamasıyla ara nokta kontrol kipini açar ya da kapatır
  - Dense root: açıksa, ara noktalardan ara değerleme (interpolation) ile yoğun yörünge kısıtları üretilir
  - Waypoint interval: geçerli kare ile ara nokta anahtar karesi arasındaki zaman aralığı, kare sayısıyla ölçülür.
- **Target Velocity** (klavyeyle hız kontrolü; **Over-the-shoulder** otomatik kamerası açıkken daha iyi sonuç verir)
  - Enable onay kutusu: hedef hız kontrolünü açar.
  - Target root velocity: XZ düzleminde hedef kök hızı.
  - Use target heading: açıksa kök konumu ve yönelimi kısıt olarak kullanılır. Kapalıysa yalnız kök konumu kısıt olarak kullanılır.
- **Post Process**
  - Enable Post-Processing: ayak kaymasını (foot skating) azaltmak ve kısıtlara daha iyi uymak için isteğe bağlı hareket düzeltmesi uygular (daha yavaştır, varsayılan olarak kapalıdır).
  - Root Margin: kök konumu düzeltmesi için pay.
  - Contact Threshold: ayak teması algılama eşiği.

#### Sekme: Visualize

- **Show Foot Contacts:** modelin ayağı yerle temasta öngördüğü karelerde ayak eklemlerini mora boyar. Yalnız **Show Skeleton** açıkken görünür.
- **Show Hand+Foot Orientations:** hem üretilen hareket hem de zaman çizelgesindeki uç işlevci anahtar kareleri için el, bilek ve ayak eklemlerinin üzerine dönme ekseni gizmoları çizer. Yönelim kısıtlarına uyulduğunu doğrulamak için işe yarar.
- **Show Reference Motion:** **Motion file path** alanında belirtilen referans hareketi kırmızı, yarı saydam bir ağ (mesh) karakter olarak gösterir.
- **Hide Distant Constraints:** açıksa, **Future Crop Length** ile belirlenen gelecek kırpma aralığının dışındaki kısıtlar gizlenir.
- **Auto Camera:** geçerli karenin insan gövdesi pozuna göre güncellenen üç otomatik kamera türünden birini seçtiğiniz açılır liste.

#### Sekme: Model

- **Acceleration Mode:** TensorRT, torch.compile ya da hiçbirini seçin.
- **Text Encoder:** LLM2Vec metin kodlayıcısının aygıtını ve duyarlığını belirler. Seçenekler: `cuda / bfloat16` (varsayılan, en hızlı, ~14 GB VRAM), `cuda / float32` (yüksek VRAM ister), `cpu / bfloat16`, `cpu / float32` (istem kodlaması daha yavaştır ama daha az VRAM gerekir).
- **TRT Export Settings:** en küçük, en uygun ve en büyük jeton sayıları ile dışa aktarma düğmesi. Bu ayarlar, gürültü giderici (denoiser) ve çözücü (decoder) için TensorRT motorlarının dışa aktarımını yapılandırır ve üretim hızını biraz etkileyebilir.
- **History Crop Length:** modele verilen en fazla geçmiş kare sayısı (en az: yama boyutu, en çok: FPS x 10 sn - üretim ufku, adım: yama boyutu, varsayılan: en az). Kısa geçmiş, yeni istemlere ve kinematik kısıtlara daha hızlı uyum sağlar. Uzun geçmiş ise daha uzun bağlam tanır, bu da karmaşık anlamsal hareket üretimine ve daha yumuşak geçişlere yardım eder.
- **Future Crop Length:** kısıt girdisi için üretim ufkunun ötesindeki gelecek kare sayısı (en az: 0, en çok: FPS x 10 sn - üretim ufku, adım: yama boyutu, varsayılan: en çok).
- **Replan Buffer Size:** yeniden planlama tamponunda kaç kare kullanılacağı. Üretim süresinin (yeniden planlama tamponu + 1) karenin oynatma süresinden kısa olmasına dikkat edin.
- **Replan Trigger Threshold:** üretilmiş ama henüz oynatılmamış kare sayısı bu eşiğe eşit ya da altında kaldığında yeniden planlama başlar.

#### Sekme: IO

- **Session Export/Load:** oturumun tamamını (hareketi ve kısıtları ya da ikisinden birini) bir dosyaya aktarır ya da dosyadan yükler.
- **Root Constraints:** kök kısıtlarını bir dosyadan yükler ya da dosyaya kaydeder.
- **Scene:** bir sahne dosyası yükler, ağın dönüşümünü ve ötelemesini ayarlar.
- **Capture Viewport Image:** geçerli görüntü alanının resmini yakalar ve bir dosyaya kaydeder.

</details>

### Demoda model entegrasyonu

Etkileşimli demo, ARDY'nin gerçek zamanlı uygulamalarda otoregresif olarak nasıl kullanılacağına iyi bir başvuru örneğidir. Ana giriş noktaları:

- **Yükleme:** [`load_model()`](https://github.com/nv-tlabs/ardy/blob/main/ardy/model/load_model.py) bir kontrol noktasını çözümler, modeli kurar ve LLM2Vec metin kodlayıcısını bağlar. Metin kodlayıcı, modeller arasında yeniden kullanılabilsin diye `load_text_encoder()` ile ayrı kurulur. Modellere takma adla başvurulur: bir iskelet adı (`"core"`, `"g1"`, `"soma"`; o iskeletin varsayılan ufkuna çözülür) ya da iskelet ve ufuk birlikte (`"core8"`, `"g152"`, `"soma60"`; yumuşak köklü SOMA çeşidi `"soma-kimodo"`). Tam ad da kullanılabilir (`"ARDY-Core-RP-20FPS-Horizon40"`). Ayrıntı için [`ardy/model/registry.py`](https://github.com/nv-tlabs/ardy/blob/main/ardy/model/registry.py) dosyasına bakın. Demo bunu [`ModelLoadingMixin.load_model`](https://github.com/nv-tlabs/ardy/blob/main/scripts/interactive_demo/loading.py) içinde bağlar.
- **Üretim:** model [`Ardy.autoregressive_step()`](https://github.com/nv-tlabs/ardy/blob/main/ardy/model/ardy_model.py) ile otoregresif çalışır. Girdi olarak metin gömmelerini ve isteğe bağlı kinematik kısıtları (`motion_mask`/`observed_motion`) alır, çıktı olarak hareket kareleri verir. Her adımdaki çağrıyı [`GenerationMixin._generate_step`](https://github.com/nv-tlabs/ardy/blob/main/scripts/interactive_demo/generation.py) içinde görebilirsiniz. Bu fonksiyon metin istemini kodlamayı, geçmişi ve kısıt maskelerini bir araya getirmeyi ve çıktıyı `motion_rep.inverse(...)` ile eklem konumlarına geri çözmeyi gösterir.

---

## 💻 Komut satırından üretim

Etkileşimsiz ya da toplu üretim için [`scripts/generate.py`](https://github.com/nv-tlabs/ardy/blob/main/scripts/generate.py) bir metin isteminden hareket üretir ve diske yazar (`.npz`; G1 için ek olarak MuJoCo-qpos `.csv`). Üretilen dosyalar varsayılan olarak `outputs/` klasörüne yazılır.

```bash
# En basiti: SOMA modeliyle 5 sn hareket, outputs/output.npz dosyasına yazılır
python scripts/generate.py "A person walks in a circle." --model soma

# Süre seçin ve tekrarlanabilir çıktı için tohumu sabitleyin (outputs/jump.npz yazılır)
python scripts/generate.py "A person jumps." --model soma --duration 8.0 --seed 0 --output jump

# Birden çok örnek (outputs/waves/ klasörüne waves_00.npz, waves_01.npz, ... yazılır)
python scripts/generate.py "A person waves." --model g1 --num_samples 4 --output waves
```

Her `.npz` dosyası üretilen hareketi taşır: `posed_joints` (dünya uzayında eklem konumları, `[T, J, 3]`), yerel ve küresel eklem dönüşleri, kök konumları ve ayak temasları. Bunlara ek olarak `fps` değeri ve istem metni `text` de dosyadadır.

Üretilen bir hareketi tarayıcıda oynatmak için (etkileşimli demoyla aynı viser görüntü alanı) [`scripts/visualize.py`](https://github.com/nv-tlabs/ardy/blob/main/scripts/visualize.py) betiğini kullanın:

```bash
python scripts/visualize.py outputs/output.npz   # tek dosya
python scripts/visualize.py outputs/waves        # --num_samples > 1 ile yazılan klasör
```

ardından `http://localhost:2334` adresini açın. İskelet (core / g1 / soma) dosyadan algılanır. Görüntüleyicide oynatma ve duraklatma, kare kaydırıcısı, ağ ve iskelet anahtarları ile çok örnekli klasörler için bir örnek açılır listesi bulunur.

En kullanışlı bayraklar:

| Bayrak | Anlamı |
|------|---------|
| `--model` | Model takma adı `soma`/`soma-kimodo`/`core`/`g1` (isteğe bağlı olarak ufukla, ör. `soma60`, `core8`) ya da tam klasör adı. Varsayılan: `soma`. |
| `--duration` | Saniye cinsinden uzunluk (varsayılan 5.0). |
| `--num_samples` | Örnek sayısı; 1'den büyükse numaralı bir klasör yazılır. |
| `--seed` | Tekrarlanabilir çıktı için rastgele tohumu sabitler. |
| `--output` | Çıktı kökü (tek örnek → tek dosya; birden çok → klasör). Yalın adlar `outputs/` altına gider. Başka yere yazmak için bir yol verin (ör. `results/jump`). |
| `--history_frames` | Her otoregresif adımın gördüğü geçmiş. Varsayılan: modelin eğitildiği 10 sn'lik pencereye sığan en uzun geçmiş. İsteme daha hızlı uyum istemiyorsanız bu değeri koruyun. |
| `--constraints` | Kaydedilmiş bir kinematik kısıt listesinin yolu. |
| `--no-postprocess` | Ayak kaymasını azaltan ve kısıtları tutturan hareket son işlemesini uygulamaz. |

> İpucu: art arda üretim yapacaksanız önce bağımsız metin kodlayıcı hizmetini arka planda başlatın (`python scripts/run_text_encoder_server.py`). Böylece her çalıştırma, LLM2Vec modelini süreç içinde her seferinde yüklemek yerine bu hizmete bağlanır.

---

## 📊 Hareket üretimi kıyaslaması (benchmark)

ARDY-SOMA-RP-30FPS-Horizon60, [Kimodo Hareket Üretimi Kıyaslaması](https://github.com/nv-tlabs/kimodo#kimodo-motion-generation-benchmark) (Kimodo Motion Generation Benchmark) üzerinde değerlendirildi. Test paketi Ardy hareket gösterimi için kuruldu (Kimodo deposunda `create_benchmark.py --motion_rep ardy`). Bu kurulumda gerçek veri (Ground Truth) hareketleri ve kök kısıt hedefleri ham kalça yörüngesini kullanır, bu yüzden Ground Truth kök hatası yapı gereği 0'dır. Tüm gövde anahtar kareleri yalnız eklem konumlarına koşullanır. Aşağıdaki tablolar, aynı Ardy hareket gösterimini tahmin eden ve aynı pakette değerlendirilen [Kimodo-SOMA-RP-ArdyRep-v1.2](https://github.com/nv-tlabs/kimodo) modelini de içerir. FID, üretilen hareketler ile Ground Truth hareketleri arasında hesaplanır. Skate cm/sn, kısıt hataları cm cinsindendir.

### Değerlendirmeyi yeniden üretme

1. **Test paketini kurun** ([Kimodo deposunda](https://github.com/nv-tlabs/kimodo), BONES-SEED'in yerel bir kopyasıyla):
   ```bash
   python benchmark/create_benchmark.py path/to/testsuite_ardy \
       --dataset datasets/bones-seed/soma_uniform --motion_rep ardy
   ```
2. **ARDY ile üretin** (bu depoda):
   ```bash
   python scripts/generate_eval.py --benchmark path/to/testsuite_ardy --model soma
   ```
3. **Gömün ve değerlendirin** (yeniden Kimodo deposunda):
   ```bash
   python benchmark/embed_folder.py path/to/testsuite_ardy
   python benchmark/evaluate_folder.py path/to/testsuite_ardy
   python benchmark/parse_folder.py path/to/testsuite_ardy
   ```

**Protokol:** aşağıdaki satırlar toplu iş boyutu 32 ve fp32 LLM2Vec metin kodlayıcısıyla, tam fp32 matris çarpımlarıyla üretildi (TF32 kapalı; pip ile kurulan PyTorch'un varsayılanı budur ve `generate_eval.py` bunu artık zorunlu kılar). [Kimodo kıyaslama sonuçları](https://github.com/nv-tlabs/kimodo/blob/main/docs/source/benchmark/results.md) ise yayımlanan protokolü kullanır (vaka başına tohumlamayla toplu iş boyutu 1, `bfloat16` metin kodlayıcı). İki protokol arasında R@3 yaklaşık +/-1 kayar, Skate ve FID ise kararlı kalır. Bu yüzden satırlar bu ayrıntı düzeyinde iki belge arasında karşılaştırılabilir.

Tablolardaki adlar Kimodo kıyaslamasının bölümleridir: **Content** ve **Repetition** test paketinin iki bölümünü, **Overview**, **Timeline single** ve **Timeline multi** metin istemi senaryolarını gösterir. **Ground Truth** gerçek veri satırıdır. ↑ yüksek değerin, ↓ düşük değerin daha iyi olduğunu belirtir.

### Metin izleme değerlendirmesi (Text-Following Evaluation)

|  | Overview R@3↑ | Overview FID↓ | Overview Skate↓ | Overview Contact↑ | Timeline single R@3↑ | Timeline single FID↓ | Timeline single Skate↓ | Timeline single Contact↑ | Timeline multi R@3↑ | Timeline multi FID↓ | Timeline multi Skate↓ | Timeline multi Contact↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Content** Ground Truth | 88.55 | 0.000 | 1.849 | 1.000 | 86.26 | 0.000 | 1.789 | 1.000 | 88.21 | 0.000 | 1.711 | 1.000 |
| **Content** ARDY-SOMA-RP-30FPS-Horizon60 | 77.86 | 0.058 | 2.760 | 0.982 | 63.90 | 0.045 | 2.609 | 0.985 | 76.43 | 0.046 | 2.452 | 0.986 |
| **Content** Kimodo-SOMA-RP-ArdyRep-v1.2 | 82.12 | 0.033 | 3.625 | 0.980 | 73.06 | 0.035 | 3.404 | 0.984 | 83.78 | 0.034 | 3.172 | 0.983 |
| **Content** ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60 | 75.57 | 0.066 | 2.230 | 0.984 | 69.68 | 0.040 | 2.215 | 0.985 | 76.68 | 0.051 | 2.052 | 0.987 |
| **Content** Kimodo-SOMA-RP-v1.2 (smooth-root suite) | 79.17 | 0.036 | 3.338 | 0.982 | 73.61 | 0.033 | 3.171 | 0.985 | 81.75 | 0.038 | 2.954 | 0.985 |
| **Repetition** Ground Truth | 93.91 | 0.000 | 2.106 | 1.000 | 89.96 | 0.000 | 2.037 | 1.000 | 94.38 | 0.000 | 1.931 | 1.000 |
| **Repetition** ARDY-SOMA-RP-30FPS-Horizon60 | 80.50 | 0.023 | 3.444 | 0.975 | 64.50 | 0.025 | 3.366 | 0.979 | 81.28 | 0.020 | 3.243 | 0.978 |
| **Repetition** Kimodo-SOMA-RP-ArdyRep-v1.2 | 85.76 | 0.009 | 4.145 | 0.973 | 74.96 | 0.011 | 3.869 | 0.977 | 87.69 | 0.012 | 3.680 | 0.976 |
| **Repetition** ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60 | 80.50 | 0.026 | 2.663 | 0.978 | 64.87 | 0.024 | 2.583 | 0.981 | 81.56 | 0.020 | 2.422 | 0.981 |
| **Repetition** Kimodo-SOMA-RP-v1.2 (smooth-root suite) | 85.46 | 0.010 | 3.927 | 0.975 | 75.17 | 0.013 | 3.701 | 0.979 | 86.73 | 0.012 | 3.554 | 0.978 |

### Kısıtlı değerlendirme (Constrained Evaluation)

> KimodoRep ve Kimodo-SOMA-RP-v1.2 (smooth-root suite) satırları, `--motion_rep kimodo` ile kurulan pakette değerlendirildi. Bu pakette kök kısıtları yumuşatılmış kökü hedefler ve Ground Truth'un kendisi de ~3.6-3.9 cm puan alır. Öteki satırlar ardy-rep paketini kullanır (ham kalça hedefleri, Ground Truth yapı gereği 0). Bu yüzden 2D Root ve Pelvis sütunlarını yalnız aynı paketteki satırlar arasında karşılaştırın.

Sütunlarda **With text** metin istemli, **Without text** metin istemsiz çalıştırmayı; **FB Pos** tüm gövde (full-body) konum hatasını, **EE Pos** uç işlevci konum hatasını gösterir.

|  | With text FB Pos↓ | With text EE Pos↓ | With text 2D Root↓ | With text Pelvis@95% | Without text FB Pos↓ | Without text EE Pos↓ | Without text 2D Root↓ | Without text Pelvis@95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Content** ARDY-SOMA-RP-30FPS-Horizon60 | 2.77 | 2.73 | 1.47 | 2.83 | 2.69 | 2.53 | 1.32 | 2.49 |
| **Content** Kimodo-SOMA-RP-ArdyRep-v1.2 | 3.08 | 3.43 | 2.90 | 6.56 | 3.08 | 3.35 | 2.64 | 5.98 |
| **Content** ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60 (smooth-root suite) | 2.81 | 2.91 | 4.08 | 6.23 | 2.67 | 2.65 | 3.97 | 6.16 |
| **Content** Kimodo-SOMA-RP-v1.2 (smooth-root suite) | 2.93 | 3.20 | 4.64 | 7.95 | 2.96 | 3.12 | 4.48 | 7.62 |
| **Repetition** ARDY-SOMA-RP-30FPS-Horizon60 | 2.78 | 2.87 | 1.54 | 3.22 | 2.63 | 2.58 | 1.26 | 2.52 |
| **Repetition** Kimodo-SOMA-RP-ArdyRep-v1.2 | 3.06 | 3.44 | 2.87 | 7.33 | 3.02 | 3.25 | 2.43 | 5.49 |
| **Repetition** ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60 (smooth-root suite) | 2.81 | 3.02 | 3.82 | 6.14 | 2.64 | 2.72 | 3.66 | 5.95 |
| **Repetition** Kimodo-SOMA-RP-v1.2 (smooth-root suite) | 2.95 | 3.22 | 4.44 | 8.22 | 2.91 | 3.17 | 4.10 | 7.41 |

---

## 📚 Belgeler

ARDY'nin belgeleri ağırlıklı olarak README dosyasındadır. README'nin tam çevirisi bu sayfadır. Hareket düzeltme kütüphanesinin kurulum notları ayrı bir belge olarak çevrildi.

| Türkçe belge | Kaynak dosya |
| --- | --- |
| Bu sayfa (README.md) | [`README.md`](https://github.com/nv-tlabs/ardy/blob/main/README.md) |
| [docs/HAREKET-DUZELTME.md](docs/HAREKET-DUZELTME.md) | [`MotionCorrection/README.md`](https://github.com/nv-tlabs/ardy/blob/main/MotionCorrection/README.md) |

Orijinal depodaki öteki metin dosyaları:

- [`CONTRIBUTING.MD`](https://github.com/nv-tlabs/ardy/blob/main/CONTRIBUTING.MD): katkı kuralları. Proje katkıları yalnız Apache-2.0 lisansıyla kabul eder ve commit'lerin DCO imzası (`git commit -s`) taşımasını ister.
- [`ATTRIBUTIONS.MD`](https://github.com/nv-tlabs/ardy/blob/main/ATTRIBUTIONS.MD): üçüncü taraf lisans bildirimleri (ör. LLM2Vec, MIT lisansı).
- [`ardy/model/llm2vec/README.md`](https://github.com/nv-tlabs/ardy/blob/main/ardy/model/llm2vec/README.md): özgün [LLM2Vec](https://github.com/McGill-NLP/llm2vec) kod tabanının, `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` modelinin `transformers==5.0.0rc3` ile çalışması için yamalanmış sürümü olduğunu açıklar.

## 🤝 NVIDIA'daki ilgili insansı çalışmaları

ARDY, NVIDIA'nın insansı hareket ve kontrol üzerine yürüttüğü birkaç çalışmayla yakından ilişkilidir:
* [Kimodo](https://github.com/nv-tlabs/kimodo) ([Türkçe repo](https://github.com/Unitree-Turkiye/nvidia-kimodo-turkce)): denetlenebilir _çevrim dışı_ hareket hazırlamada uzmanlaşmış insan hareketi üretim modeli.
* [MotionBricks](https://nvlabs.github.io/motionbricks/) ([Türkçe repo](https://github.com/Unitree-Turkiye/unitree-g1-motionbricks-turkce)): hızlı ve sağlam hareket ara kare üretiminde (inbetweening) uzmanlaşmış gerçek zamanlı hareket üretim çatısı.
* [GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) ([Türkçe repo](https://github.com/Unitree-Turkiye/unitree-g1-gear-sonic-turkce)): fiziksel robotlar için insansı davranış temel modeli. ARDY, SONIC'e verilecek hareketleri üreten bir planlayıcı olarak kullanılabilir.

---

## 🚀 Unitree G1 ile ARDY: Türkiye'de nereden başlanır

ARDY'nin iki kontrol noktası doğrudan Unitree G1 iskeleti için eğitildi: `ARDY-G1-RP-25FPS-Horizon52` ve `ARDY-G1-RP-25FPS-Horizon8`. İkisi de Bones Rigplay 1 verisiyle 25 FPS'de üretir ve komut satırında `--model g1` (ya da `g152`) takma adıyla çağrılır. `scripts/generate.py`, G1 için `.npz` dosyasının yanına MuJoCo-qpos biçiminde bir `.csv` dosyası da yazar. Kinematik kısıt örneklemesi Bones SEED veri setinin G1 CSV hareketleriyle çalışır. Orijinal depo G1 iskelet varlıklarını `ardy/assets/skeletons/g1skel34` klasöründe taşır.

README'ye göre ARDY, fiziksel insansı robotlar için davranış temel modeli olan GEAR SONIC'e verilecek hareketleri üreten bir planlayıcı olarak kullanılabilir. ARDY'nin çıktısı kinematik harekettir. Fiziksel robota geçişte izleme kontrolcüsü, eklem sınırları ve güvenlik doğrulaması sizin iş akışınızın parçasıdır.

**Platform seçimi**

| Ürün | ARDY iş akışındaki yeri |
| --- | --- |
| [Unitree G1](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1) | ARDY G1 modellerinin hedeflediği insansı robot, 132 cm, 35 kg, 23-43 DoF |
| [Unitree G1 EDU teklif](https://www.robotlar.org/tr/fiyat/unitree-g1-edu-fiyat) | Geliştirici ve araştırma sürümü için teklif formu |
| [Unitree G1 Plus](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1-plus) | 2 DoF boyun, yüksek tork, dokunsal kafa ve 6 mikrofonlu G1 |
| [Unitree G1-D](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1-d) | İnsansı robot veri toplama ve eğitim platformu, Jetson Orin |
| [G1 Ar-Ge platformu](https://www.robotlar.org/tr/g1-arge-platformu) | Üniversite ve Ar-Ge laboratuvarları için G1 çözüm sayfası |

**Türkçe rehberler (iş akışına göre)**

- Bu deponun özeti ve G1 modelleri: [NVIDIA ARDY gerçek zamanlı hareket rehberi](https://www.robotlar.org/tr/rehber/nvidia-ardy-gercek-zamanli-hareket-rehberi)
- Çevrim dışı hareket hazırlama ve ARDY'nin değerlendirildiği kıyaslama: [NVIDIA Kimodo hareket üretimi rehberi](https://www.robotlar.org/tr/rehber/nvidia-kimodo-hareket-uretimi-rehberi)
- Bağlam ve hedef pozlar arasında gerçek zamanlı ara kare üretimi: [Unitree G1 MotionBricks rehberi](https://www.robotlar.org/tr/rehber/unitree-g1-motionbricks-rehberi)
- Üretilen hareketi gerçek G1'de tüm gövde kontrolüyle izletme: [GEAR SONIC Unitree G1 tüm gövde kontrol rehberi](https://www.robotlar.org/tr/rehber/gear-sonic-unitree-g1-tum-govde-kontrol-rehberi)
- Kısıt örneklemesinde kullanılan G1 hareket verisi: [BONES-SEED hareket veri seti rehberi](https://www.robotlar.org/tr/rehber/bones-seed-hareket-veri-seti-rehberi)
- Isaac GR00T akışı: [Unitree G1 Isaac GR00T rehberi](https://www.robotlar.org/tr/rehber/unitree-g1-isaac-gr00t-rehberi)
- Referans hareketi izleyen politikaları eğitmek için pekiştirmeli öğrenme: [Unitree RL Gym](https://www.robotlar.org/unitree/rl-gym)
- Robotla haberleşme ve düşük seviye kontrol: [Unitree SDK rehberi](https://www.robotlar.org/unitree/sdk-rehberi)

**Türkçe GitHub depoları:** [G1](https://github.com/Unitree-Turkiye/unitree-g1-turkce) · [SDK2](https://github.com/Unitree-Turkiye/unitree-sdk2-turkce) · [G1 MotionBricks](https://github.com/Unitree-Turkiye/unitree-g1-motionbricks-turkce) · [Kimodo](https://github.com/Unitree-Turkiye/nvidia-kimodo-turkce) · [GEAR SONIC](https://github.com/Unitree-Turkiye/unitree-g1-gear-sonic-turkce)

Üniversite ve kurumlar için platform, eğitim ve Türkçe geliştirici desteği konusunda [kurumsal teklif formundan](https://www.robotlar.org/tr/fiyat/unitree-kurumsal-teklif-al) bize ulaşabilirsiniz.

## 🧬 NVIDIA İnsansı Hareket Ailesi (Türkçe)

NVIDIA'nın animasyon, robotik ve fiziksel yapay zeka için yayımladığı insansı hareket projeleri birbirini tamamlar: SOMA ortak gövde modelini, BONES-SEED veriyi, GEM videodan hareketi, Kimodo ve ARDY hareket üretimini, MotionBricks gerçek zamanlı hareket planlamayı, SOMA Retargeter insan hareketini Unitree G1'e aktarmayı, ProtoMotions fizik tabanlı eğitimi, GEAR SONIC de gerçek robotta tüm gövde kontrolünü üstlenir. Akışın bütününü [merkez rehberde](https://www.robotlar.org/tr/rehber/nvidia-insansi-hareket-ekosistemi-unitree-g1-rehberi) anlatıyoruz.

| Proje | Ne yapar | Türkçe repo | Türkçe rehber | Resmi depo |
|---|---|---|---|---|
| MotionBricks | Bağlam ve hedef pozlar arasında gerçek zamanlı hareket üretir, Unitree G1 modeliyle gelir | [unitree-g1-motionbricks-turkce](https://github.com/Unitree-Turkiye/unitree-g1-motionbricks-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/unitree-g1-motionbricks-rehberi) | [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) |
| Kimodo | Denetlenebilir çevrim dışı insan hareketi üretimi | [nvidia-kimodo-turkce](https://github.com/Unitree-Turkiye/nvidia-kimodo-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/nvidia-kimodo-hareket-uretimi-rehberi) | [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) |
| **ARDY** | **Kimodo üzerine gerçek zamanlı, etkileşimli hareket üretimi** | **[nvidia-ardy-turkce](https://github.com/Unitree-Turkiye/nvidia-ardy-turkce)** | **[Rehber](https://www.robotlar.org/tr/rehber/nvidia-ardy-gercek-zamanli-hareket-rehberi)** | **[nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)** |
| SOMA Body Model | Birleşik, parametrik insan gövde modeli | [nvidia-soma-govde-modeli-turkce](https://github.com/Unitree-Turkiye/nvidia-soma-govde-modeli-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/nvidia-soma-govde-modeli-rehberi) | [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) |
| BONES-SEED | SOMA ve G1 biçiminde büyük hareket yakalama veri seti | [bones-seed-hareket-veri-seti-turkce](https://github.com/Unitree-Turkiye/bones-seed-hareket-veri-seti-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/bones-seed-hareket-veri-seti-rehberi) | [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed) |
| ProtoMotions | Benzetilen insan ve insansıları eğitmek için GPU hızlandırmalı simülasyon çatısı | [nvidia-protomotions-turkce](https://github.com/Unitree-Turkiye/nvidia-protomotions-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/nvidia-protomotions-insansi-simulasyon-rehberi) | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) |
| SOMA Retargeter | SOMA hareketini Unitree G1 gibi insansı robotlara aktarır | [unitree-g1-soma-retargeter-turkce](https://github.com/Unitree-Turkiye/unitree-g1-soma-retargeter-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/soma-retargeter-unitree-g1-hareket-aktarimi-rehberi) | [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) |
| GEM | Görüntü ve videodan tüm gövde ve el hareketini kestirir | [nvidia-gem-x-turkce](https://github.com/Unitree-Turkiye/nvidia-gem-x-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/nvidia-gem-x-videodan-hareket-rehberi) | [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) |
| GEAR SONIC | Fiziksel insansı robotlar için davranış temel modeli ve tüm gövde kontrolü | [unitree-g1-gear-sonic-turkce](https://github.com/Unitree-Turkiye/unitree-g1-gear-sonic-turkce) | [Rehber](https://www.robotlar.org/tr/rehber/gear-sonic-unitree-g1-tum-govde-kontrol-rehberi) | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) |

İlgili Türkçe kaynaklar: [Unitree G1 + Isaac GR00T](https://github.com/Unitree-Turkiye/unitree-g1-isaac-groot-turkce) · [XR teleoperasyon](https://github.com/Unitree-Turkiye/unitree-teleoperasyon-turkce) · [LeRobot](https://github.com/Unitree-Turkiye/unitree-lerobot-turkce) · [RL Gym](https://github.com/Unitree-Turkiye/unitree-rl-gym-turkce) · [MuJoCo](https://github.com/Unitree-Turkiye/unitree-mujoco-turkce)

## 🤖 Unitree Ürün Ailesi (Türkiye)

Robotlar.org, Unitree Robotics'in Türkiye resmi distribütörüdür. Aşağıdaki ürünlerin Türkçe tanıtımı, teknik özellikleri ve kurumsal teklif formu sitede yer alır. Fiyatlar konfigürasyona göre değiştiği için [teklif üzerinden](https://www.robotlar.org/tr/fiyat/unitree-kurumsal-teklif-al) paylaşılır.

### İnsansı robotlar

| Ürün | Ne işe yarar |
|---|---|
| [Unitree G1](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1) | 132 cm, 35 kg, 23-43 DoF insansı robot |
| [Unitree G1-D](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1-d) | İnsansı robot veri toplama ve eğitim platformu, Jetson Orin |
| [Unitree G1 Plus](https://www.robotlar.org/tr/insansi-robotlar/unitree-g1-plus) | 2 DoF boyun, yüksek tork, dokunsal kafa ve 6 mikrofonlu G1 |
| [Unitree H2](https://www.robotlar.org/tr/insansi-robotlar/unitree-h2) | 182 cm, 70 kg, 31 DoF tam boy insansı robot |
| [Unitree H2-D](https://www.robotlar.org/tr/insansi-robotlar/unitree-h2-d) | 7 DoF kollu, tekerlekli çift kollu insansı robot |
| [Unitree H2 Plus](https://www.robotlar.org/tr/insansi-robotlar/unitree-h2-plus) | NVIDIA Isaac GR00T referans insansı robotu, 75 DoF, Jetson Thor |
| [Unitree H2-X](https://www.robotlar.org/tr/insansi-robotlar/unitree-h2-x) | Hizmet ve koruma görevleri, 4 m platformdan iniş |
| [Unitree R1](https://www.robotlar.org/tr/insansi-robotlar/unitree-r1) | 123 cm, 29 kg, 26 DoF insansı robot (AIR, R1, EDU sürümleri) |
| [Unitree R1-D](https://www.robotlar.org/tr/insansi-robotlar/unitree-r1-d) | R1-A5 ya da R1-A7 (7-DOF) kollu çift kollu insansı platform |
| [Tüm insansı robotlar →](https://www.robotlar.org/tr/insansi-robotlar) | |

### Robot elleri

| Ürün | Ne işe yarar |
|---|---|
| [Unitree Dex3-1](https://www.robotlar.org/tr/robot-elleri/unitree-dex3-1) | G1 için 7 serbestlik dereceli el, opsiyonel 33 dokunsal sensör |
| [Unitree Dex5-1](https://www.robotlar.org/tr/robot-elleri/unitree-dex5-1) | 20 serbestlik dereceli (16 aktif) el, Dex5-1P sürümünde 94 dokunsal sensör |
| [Unitree Dex5-S](https://www.robotlar.org/tr/robot-elleri/unitree-dex5-s) | 22 aktif serbestlik, 620 g, çift T1s Ethernet, 1000 Hz |
| [Unitree Dex1-1](https://www.robotlar.org/tr/robot-elleri/unitree-dex1-1) | İki çeneli tutucu: 90 mm strok, 5-120 N ayarlı kavrama, RS-485 |
| [Tüm robot elleri →](https://www.robotlar.org/tr/robot-elleri) | |

### Dört ayaklı robotlar ve bileşenler

[Go2](https://www.robotlar.org/tr/kopek-robotlar/unitree-go2) · [Go2-W](https://www.robotlar.org/tr/kopek-robotlar/unitree-go2-w) · [B2](https://www.robotlar.org/tr/kopek-robotlar/unitree-b2) · [B2-W](https://www.robotlar.org/tr/kopek-robotlar/unitree-b2-w) · [A2](https://www.robotlar.org/tr/kopek-robotlar/unitree-a2) · [AS2](https://www.robotlar.org/tr/kopek-robotlar/unitree-as2) · [Tüm dört ayaklı robotlar](https://www.robotlar.org/tr/kopek-robotlar) · [4D LiDAR L2](https://www.robotlar.org/tr/unitree-bilesenler/4d-lidar-l2) · [Tüm Unitree bileşenleri](https://www.robotlar.org/tr/unitree-bilesenler)

## 📖 Türkçe Rehberler

- [NVIDIA insansı hareket projeleri ve Unitree G1 (merkez rehber)](https://www.robotlar.org/tr/rehber/nvidia-insansi-hareket-ekosistemi-unitree-g1-rehberi)
- [NVIDIA ARDY gerçek zamanlı hareket rehberi](https://www.robotlar.org/tr/rehber/nvidia-ardy-gercek-zamanli-hareket-rehberi)
- [NVIDIA Kimodo hareket üretimi rehberi](https://www.robotlar.org/tr/rehber/nvidia-kimodo-hareket-uretimi-rehberi)
- [Unitree G1 MotionBricks rehberi](https://www.robotlar.org/tr/rehber/unitree-g1-motionbricks-rehberi)
- [GEAR SONIC Unitree G1 tüm gövde kontrol rehberi](https://www.robotlar.org/tr/rehber/gear-sonic-unitree-g1-tum-govde-kontrol-rehberi)
- [Unitree G1 Isaac GR00T rehberi](https://www.robotlar.org/tr/rehber/unitree-g1-isaac-gr00t-rehberi)
- [Unitree LeRobot: veri toplama ve taklit öğrenme](https://www.robotlar.org/tr/rehber/unitree-lerobot-rehberi)
- [Unitree G1 XR teleoperasyon](https://www.robotlar.org/unitree/g1-teleoperasyon)
- [Unitree RL Gym](https://www.robotlar.org/unitree/rl-gym)
- [Unitree SDK rehberi](https://www.robotlar.org/unitree/sdk-rehberi)
- [Unitree eğitim çözümleri (üniversite ve laboratuvar)](https://www.robotlar.org/tr/unitree-egitim-cozumleri)
- [Tüm açık kaynak Unitree projeleri (Türkçe)](https://www.robotlar.org/tr/acik-kaynak)

## 🔗 Unitree Türkiye GitHub Ağı

**Modeller:** [Go2](https://github.com/Unitree-Turkiye/unitree-go2-turkce) · [Go2-W](https://github.com/Unitree-Turkiye/unitree-go2-w-turkce) · [B2](https://github.com/Unitree-Turkiye/unitree-b2-turkce) · [A2](https://github.com/Unitree-Turkiye/unitree-a2-turkce) · [G1](https://github.com/Unitree-Turkiye/unitree-g1-turkce) · [H2](https://github.com/Unitree-Turkiye/unitree-h2)
**Geliştirici:** [SDK2](https://github.com/Unitree-Turkiye/unitree-sdk2-turkce) · [XR Teleoperasyon](https://github.com/Unitree-Turkiye/unitree-teleoperasyon-turkce) · [LeRobot](https://github.com/Unitree-Turkiye/unitree-lerobot-turkce) · [Becerikli El](https://github.com/Unitree-Turkiye/unitree-hunerli-el-turkce) · [MuJoCo](https://github.com/Unitree-Turkiye/unitree-mujoco-turkce) · [ROS2](https://github.com/Unitree-Turkiye/unitree-ros2-turkce) · [VLA](https://github.com/Unitree-Turkiye/unitree-vla-turkce)
**G1 araştırma:** [G1 MotionBricks](https://github.com/Unitree-Turkiye/unitree-g1-motionbricks-turkce) · [ARDY](https://github.com/Unitree-Turkiye/nvidia-ardy-turkce) · [Kimodo](https://github.com/Unitree-Turkiye/nvidia-kimodo-turkce) · [G1 dokunsal algılama giysisi](https://github.com/Unitree-Turkiye/unitree-g1-dokunsal-giysi-turkce) · [🏠 Unitree Türkiye](https://github.com/Unitree-Turkiye)

## 🙏 Teşekkür

Orijinal proje şu açık kaynak projelerin üzerine kuruludur (orijinal depodan):

- Etkileşimli 3B hareket üretimi demosu için [Viser](https://github.com/nerfstudio-project/viser)
- Metin kodlama için [LLM2Vec](https://github.com/McGill-NLP/llm2vec)

## 📄 Lisans

Orijinal kod tabanı [Apache-2.0](https://github.com/nv-tlabs/ardy/blob/main/LICENSE) ile lisanslıdır. Model kontrol noktaları ve veriler, Hugging Face indirme sayfalarında belirtildiği gibi ayrı lisanslarla gelir. README'deki kontrol noktası tablosu modeller için [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) lisansını gösterir.

Proje, ek üçüncü taraf açık kaynak yazılım projelerini indirir ve kurar. Kullanmadan önce bu projelerin lisans koşullarını inceleyin. Üçüncü taraf bildirimleri [ATTRIBUTIONS.MD](https://github.com/nv-tlabs/ardy/blob/main/ATTRIBUTIONS.MD) dosyasındadır.

**Bu çevirinin lisansı:** Bu depodaki metinler, NVIDIA'nın [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) deposundaki belgelerin Türkçe çevirisidir ve kaynağın lisansı olan [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) ile yayımlanır. Özgün eser: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Değişiklik: Türkçeye çeviri ve Türkiye bağlantıları. Bu depo kaynak kod, model, veri ya da medya dosyası barındırmaz. Görseller orijinal depodan gösterilir. Ayrıntı [LICENSE](LICENSE) dosyasında.

## 📝 Atıf

Bu kodu araştırmanızda kullanırsanız lütfen şu atfı verin:

```bibtex
@article{zhao2026ardy,
  title   = {ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation},
  author  = {Zhao, Kaifeng and Petrovich, Mathis and Zhang, Haotian and Wang, Tingwu and Tang, Siyu and Rempe, Davis},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026},
  volume  = {45},
  number  = {4},
  articleno = {86},
  doi     = {10.1145/3811284}
}
```

## ✉️ İletişim

Sorular ve sorunlar için [orijinal depoda](https://github.com/nv-tlabs/ardy/issues) bir issue açın ya da doğrudan yazarlara ulaşın. Bu Türkçe çeviriyle ilgili düzeltme önerilerinizi bu depoda issue olarak iletebilirsiniz.

---

> ℹ️ **Unitree Türkiye:** Unitree Robotics'in Türkiye resmi satış ve teknik distribütörü. Unitree G1 ailesi için teklif, kurulum, eğitim ve Türkçe geliştirici desteği: **[robotlar.org](https://www.robotlar.org/tr/rehber/unitree-turkiye-distributoru)** · 📧 info@robotlar.org

© 2026 **[www.robotlar.org](https://www.robotlar.org)**
