from embedd import EmbeddingModel
from Cross import crossencoder
from mmr import MMRSelector
from textrank import textrank_selection
from preprocess import VietnamesePreprocessor

class summaries:
    def __init__(self, documents):
        
        self.documents = documents
    
    def get_documents(self):
        return self.documents
    
    def summarize(self, top_k=5, diversity=0.7, textrank_k=20):

        if not isinstance(self.documents, str):
            return "Nhập văn bản đầu vào là chuỗi ký tự."
        
        if len(self.documents.split()) >= 1500:
            return "Nhiều quá 1000 từ, vui lòng rút gọn văn bản đầu vào."
        
        if len(self.documents.strip()) == 0:
            return "Rỗng"
        
        preprc = VietnamesePreprocessor()
        self.documents = preprc.process_document(self.documents)
        
        emb_model = EmbeddingModel()
        ce_model = crossencoder()
        mmr_selector = MMRSelector(top_k=top_k, diversity=diversity)
    
        sents, embs = emb_model.embed_text(self.documents)
        

        textrank = textrank_selection(embs, textrank_k)
        rank = textrank.select_sentences()

        if 0 not in rank and len(sents) > 0:
            rank = [0] + rank[:-1]

        candidate_texts = [sents[idx] for idx in rank]
        candidate_embs = embs[rank]


        doc_positional = " ".join([
            f"[POS_{i}] {sentence}" for i, sentence in enumerate(sents)
        ])


        positional_candidates = []
        for sent, idx in zip(candidate_texts, rank):
            positional_candidates.append(f"[POS_{idx}] {sent}")
        

        doc_texts = [doc_positional] * len(positional_candidates)
        ce_scores = ce_model.rank_sentences(doc_texts, positional_candidates)


        doc_emb = emb_model.model.encode([self.documents])[0]

        selected_idx = mmr_selector.select(candidate_embs, doc_emb, ce_scores=ce_scores)
        chosen_global_idxs = [rank[i] for i in selected_idx]
        chosen_global_idxs = sorted(chosen_global_idxs)

        summary_sentences = [sents[i] for i in chosen_global_idxs]

        return ' '.join(summary_sentences)


    
if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint
    # df = pd.read_json('train_clean.jsonl', lines =True)
    # text = df.loc[0, 'text']
    
    text = """Trong thập kỷ qua, Trí tuệ Nhân tạo (AI) đã vượt ra khỏi những trang sách khoa học viễn tưởng để trở thành một phần không thể thiếu của cuộc sống hiện đại. Từ những trợ lý ảo đơn giản trên điện thoại như Siri hay Google Assistant cho đến các hệ thống phức tạp điều khiển xe tự lái, AI đang hiện diện ở khắp mọi nơi. Sự bùng nổ của các mô hình ngôn ngữ lớn như ChatGPT gần đây đã đánh dấu một bước ngoặt lịch sử, thay đổi hoàn toàn cách con người tương tác với máy móc. Cuộc cách mạng công nghệ này không chỉ mang lại những tiện ích vượt trội mà còn đặt ra những thách thức chưa từng có đối với nhân loại.

Lợi ích to lớn nhất mà AI mang lại chính là khả năng xử lý dữ liệu và tự động hóa vượt trội. Trong lĩnh vực y tế, các thuật toán học máy đang giúp các bác sĩ chẩn đoán bệnh ung thư sớm với độ chính xác cao hơn mắt thường. Việc ứng dụng công nghệ thông minh vào chăm sóc sức khỏe đang cứu sống hàng triệu người mỗi ngày. Trí tuệ nhân tạo hỗ trợ ngành y tế phân tích hình ảnh X-quang nhanh chóng, giúp giảm tải áp lực cho đội ngũ y bác sĩ. Không chỉ vậy, AI còn tham gia vào quá trình nghiên cứu và phát triển vắc-xin, rút ngắn thời gian thử nghiệm từ hàng năm xuống còn vài tháng. Rõ ràng, sự đóng góp của máy móc trong việc bảo vệ sức khỏe con người là không thể phủ nhận.

Bên cạnh y tế, lĩnh vực kinh tế và sản xuất cũng được hưởng lợi sâu sắc từ làn sóng công nghệ này. Các nhà máy thông minh sử dụng robot tự động để vận hành dây chuyền sản xuất 24/7 mà không cần nghỉ ngơi. Năng suất lao động tăng lên đáng kể nhờ sự hỗ trợ của các hệ thống tự động hóa. AI giúp tối ưu hóa quy trình làm việc, giảm thiểu sai sót của con người và tiết kiệm chi phí vận hành cho doanh nghiệp. Các thuật toán dự báo thị trường cũng giúp các nhà đầu tư đưa ra quyết định chính xác hơn trong lĩnh vực tài chính. Việc áp dụng trí tuệ nhân tạo vào sản xuất kinh doanh đang là chìa khóa để thúc đẩy tăng trưởng kinh tế toàn cầu.

Tuy nhiên, sự phát triển thần tốc của AI cũng đi kèm với những nỗi lo ngại sâu sắc về vấn đề việc làm. Nhiều chuyên gia dự báo rằng robot và các phần mềm tự động sẽ thay thế con người trong nhiều ngành nghề. Nỗi sợ hãi về việc máy móc cướp đi công việc của người lao động đang ngày càng gia tăng. Từ những công việc tay chân đơn giản đến những nghề nghiệp đòi hỏi kỹ năng như biên dịch hay lập trình, không ai là an toàn tuyệt đối trước làn sóng tự động hóa. Tỷ lệ thất nghiệp có thể gia tăng nếu xã hội không kịp thích nghi với sự thay đổi này. Viễn cảnh robot thay thế con người làm việc đang đặt ra bài toán khó cho các nhà hoạch định chính sách về an sinh xã hội.

Một thách thức khác không kém phần nghiêm trọng là vấn đề đạo đức và an ninh thông tin. Các hệ thống AI có thể vô tình học theo những định kiến sai lệch có sẵn trong dữ liệu, dẫn đến sự phân biệt đối xử. Ngoài ra, sự xuất hiện của công nghệ Deepfake cho phép tạo ra các video và âm thanh giả mạo y như thật, gây hoang mang dư luận. Tin giả và lừa đảo trực tuyến đang trở nên tinh vi hơn nhờ sự tiếp tay của trí tuệ nhân tạo. An ninh mạng đang bị đe dọa bởi những công cụ AI mạnh mẽ nằm trong tay kẻ xấu. Chúng ta cần phải cảnh giác trước những rủi ro tiềm tàng mà công nghệ này mang lại.

Tóm lại, Trí tuệ Nhân tạo là một con dao hai lưỡi. Nó vừa là công cụ đắc lực giúp nâng cao chất lượng cuộc sống, vừa là nguồn gốc của những bất ổn xã hội nếu không được kiểm soát tốt. Tương lai không nằm ở việc AI sẽ thay thế con người, mà là cách con người hợp tác với AI để tạo ra những giá trị mới. Chúng ta cần xây dựng những hành lang pháp lý vững chắc để phát triển AI một cách có trách nhiệm. Việc cân bằng giữa đổi mới công nghệ và đạo đức con người sẽ là yếu tố quyết định sự thành bại của kỷ nguyên số này. Con người cần làm chủ công nghệ thay vì để công nghệ làm chủ con người.
"""
    # print(type(text))
    # exit()
    
    summarizer = summaries(text)
    summary = summarizer.summarize(top_k=7, diversity=0.5, textrank_k=20)
    print("Tóm tắt:")
    print(summary)
        