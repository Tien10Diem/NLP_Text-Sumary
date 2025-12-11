import re
import unicodedata
from underthesea import sent_tokenize
from transformers import pipeline

class VietnamesePreprocessor:
    def __init__(self):
        # 1. Mapping sửa lỗi dấu thanh (chuẩn hóa vị trí dấu)
        self.ViTriDau = {
            "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
            "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
            "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
            "uả": "ủa", "ả": "ả", "ỏ": "ỏ", "ủ": "ủ"
        }
        # 
        self.qu = {
            r'qúa': 'quá', r'qùa': 'quà', r'qủa': 'quả', r'qũa': 'quã', r'qụa': 'quạ',
            r'qúy': 'quý', r'qùy': 'quỳ', r'qủy': 'quỷ', r'qũy': 'quỹ', r'qụy': 'quỵ',
            r'qúe': 'qué', r'qùe': 'què', r'qủe': 'quẻ', r'qũe': 'quẽ', r'qụe': 'quẹ',
        }
        # 3. Mapping chuẩn hóa i/y
        self.iy = {
            r'\bkĩ\s': 'kỹ ', r'\blí\s': 'lý ', r'\bkì\s': 'kỳ ', r'\bmĩ\s': 'mỹ ',
            r'\bhi vọng\b': 'hy vọng', r'\bqui\b': 'quy'
        }

    def normalize_text(self, text):
        if not isinstance(text, str): return ""
        
        # Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)

        for old, new in self.ViTriDau.items():
            text = text.replace(old, new)

        for wrong, right in self.qu.items():
            text = re.sub(wrong, right, text, flags=re.IGNORECASE)

        for pattern, replacement in self.iy.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # def laptu(self, text):
                
    #     corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
    #     result = corrector(text)
        
    #     return result[0]['generated_text']
    
    def process_document(self, text):

        clean_text = self.normalize_text(text)
        # clean_text = self.laptu(clean_text)
        
        return clean_text
if __name__ == "__main__":
    preprocessor = VietnamesePreprocessor()

    raw_text = """
    Qúa tuyệt vời!!! <br>
    Tôi là một kĩ sư phần mềm. Hi vọng dự án thành công.
    Liên hệ: spam@gmail.com hoặc truy cập https://website-rac.net
    
    """

    # Chạy xử lý
    clean_sentences = preprocessor.process_document(raw_text)
    print("Văn bản sau khi xử lý:")
    print(clean_sentences)