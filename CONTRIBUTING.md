# Katkıda Bulunma Rehberi

Projeye katkıda bulunmak istediğiniz için teşekkürler! Bu rehber, katkı sürecini kolaylaştırmak için hazırlanmıştır.

## İçindekiler
- [Başlamadan Önce](#başlamadan-önce)
- [Geliştirme Ortamı Kurulumu](#geliştirme-ortamı-kurulumu)
- [Kod Stileri](#kod-stileri)
- [Katkı Adımları](#katkı-adımları)
- [Commit Mesajları](#commit-mesajları)
- [Pull Request Süreci](#pull-request-süreci)
- [Test Etme](#test-etme)

## Başlamadan Önce

Katkıda bulunmadan önce:
- Projeyi fork edin
- Projeyi yerel makinenize klonlayın
- Branch kurallarımızı okuyun
- Kod Davranış Kurallarımızı okuyun

## Geliştirme Ortamı Kurulumu

### Gereksinimler
- Python 3.8+
- pip veya conda

### Ortam Kurulumu

```bash
# Projeyi klonlayın
git clone https://github.com/YOUR_USERNAME/dl_xview_yolo.git
cd dl_xview_yolo

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükleyin
pip install -r requirements.txt
pip install -e .
```

## Kod Stileri

### Python Kodlama Standartları
- **PEP 8** kurallarına uyun
- Satır uzunluğu maksimum 88 karakter (Black formatter)
- Fonksiyonlar ve sınıflar için docstring yazın
- Anlamlı değişken adları kullanın

### Kullanılan Araçlar
```bash
# Kodu formatla
black .

# Lint kontrol
flake8 .

# Type checking
mypy .
```

### Docstring Örneği
```python
def detect_objects_in_satellite_image(image_path: str, confidence: float = 0.5) -> dict:
    """
    Uydu görüntüsünde nesne tespiti yapar.
    
    Args:
        image_path (str): Uydu görüntüsünün yolu
        confidence (float): Tespit güven eşiği (0-1 arası)
    
    Returns:
        dict: Tespit sonuçları
    """
    pass
```

## Katkı Adımları

1. **Issue Oluşturun veya Bulun**
   - Bir bug buldum veya özellik önerisi mi? Önce bir issue açın
   - Varolan issue'leri kontrol edin, aynı konu üzerinde çalışılmıyor mu diye

2. **Feature Branch Oluşturun**
   ```bash
   git checkout -b feature/your-feature-name
   # veya
   git checkout -b bugfix/your-bug-name
   ```

3. **Değişiklikleri Yapın**
   - Küçük, mantıklı adımlar halinde commit yapın
   - Yalnızca ilgili dosyaları değiştirin

4. **Branch Adlandırma Kuralları**
   - Feature: `feature/descriptive-name`
   - Bug Fix: `bugfix/issue-description`
   - Documentation: `docs/description`
   - Örnek: `feature/yolov8-model-optimization`

## Commit Mesajları

Anlaşılır commit mesajları yazın:

### Örnek Formatı
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Tipleri
- `feat`: Yeni özellik
- `fix`: Hata düzeltimi
- `docs`: Dokümantasyon güncelleme
- `style`: Kod formatı değişiklikleri (PEP 8)
- `refactor`: Kod yeniden düzenleme
- `test`: Test ekleme/güncelleme
- `chore`: Diğer değişiklikler

### Örnekler
```
feat(detection): YOLOv8 model optimizasyonu ekle

fix(data_loader): xView veri yükleme hatasını düzelt

docs(readme): Kurulum talimatlarını güncelle

refactor(utils): Yardımcı fonksiyonları modülarize et
```

## Pull Request Süreci

### PR Açmadan Önce
- [ ] Ana branch'le güncellenmiş misiniz? (`git pull origin main`)
- [ ] Testler geçiyor mu?
- [ ] Kod formatı kontrol edildi mi? (`black`, `flake8`)
- [ ] Docstring ve yorumlar yazılı mı?
- [ ] CHANGELOG güncellenmiş mi?

### PR Şablonu
```markdown
## Açıklama
Kısaca ne yaptığınızı açıklayın

## İlgili Issue
Closes #issue_number

## Değişiklik Türü
- [ ] Bug fix
- [ ] Yeni özellik
- [ ] Backward incompatible değişiklik
- [ ] Dokümantasyon güncellemesi

## Test Edildi Mi?
- [ ] Lokal ortamda test edildi
- [ ] Test case'ler eklendi
- [ ] Mevcut testler hala geçiyor

## Checklist
- [ ] Kodun kendini açıklayıcı olduğundan emin misiniz?
- [ ] Gereksiz yorum kaldırılmış mı?
- [ ] Dokümantasyon güncellenmiş mi?
```

## Test Etme

### Unit Test'ler Çalıştırma
```bash
pytest tests/
```

### Belirli Test'i Çalıştırma
```bash
pytest tests/test_detection.py::test_yolov8_inference
```

### Test Kapsamı Kontrol
```bash
pytest --cov=src tests/
```

### Yeni Test Yazma
```python
# tests/test_detection.py
import unittest
from src.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ObjectDetector(model_name='yolov8n')
    
    def test_detect_objects(self):
        results = self.detector.detect('test_image.jpg')
        self.assertIsNotNone(results)
        self.assertIn('detections', results)
```

## Sık Sorulan Sorular

**S: PR'ımı nasıl güncel tutarım?**
A:
```bash
git fetch origin
git rebase origin/main
git push --force-with-lease origin your-branch
```

**S: Hata yaptığım commit'ı nasıl düzeltirim?**
A:
```bash
git commit --amend
# veya
git rebase -i HEAD~n  # son n commit'i düzenlemek için
```

**S: Branch'imi nasıl silirim?**
A:
```bash
git branch -d local-branch
git push origin --delete remote-branch
```

## İletişim

Sorularınız varsa:
- Issue açın
- Discussions sekmesini kullanın
- Proje maintainer'ına ulaşın

---

**Not:** Tüm katkılar MIT Lisansı altında kabul edilir.

Katkılarınız için teşekkürler! 🙏