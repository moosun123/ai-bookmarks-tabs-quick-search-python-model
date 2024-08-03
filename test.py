from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

 
# 初始化Flask应用
app = Flask(__name__)

def get_embedding_map(model, sentences, embedding_word):
    embedding_list = []
    embedding_map = {} 
    for sentence in sentences:
        print(sentence, 'sentence')
        embedding = model.encode(sentence)
        embedding_list.append(embedding.tolist())  # 转换为列表以便于JSON序列化
        similarity = util.pytorch_cos_sim(embedding_word, embedding).item()
        embedding_map[sentence] = similarity
    return embedding_map
    
def get_sorted_embedding_map(map):
    return  sorted(map.items(), key = lambda item: item[1], reverse=True)


@app.route('/similarity', methods=['POST'])
def compute_similarity():
    try: 
        # 获取请求中的句子
        data = request.json
        print(data, 'data')

        print(model, 'model')


        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        
        word = data.get('word')
        embedding_word = model.encode(word)
        print('embedding_word',word )

         # 生成句子嵌入
        embedding_map_tabs = get_embedding_map(model, data.get('tabs'), embedding_word)
        embedding_map_bookmarks = get_embedding_map(model, data.get('bookmarks'), embedding_word)
        print(embedding_map_tabs, embedding_map_bookmarks )
        return jsonify({
            'bookmarks': get_sorted_embedding_map(embedding_map_bookmarks), 
            'tabs':   get_sorted_embedding_map(embedding_map_tabs)
            })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
   


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# # 需要比较的两个句子
# sentence1 = "moshu"
# sentence2 = "魔数"
# sentence3 = "魔术"
# # 生成嵌入
# embedding1 = model.encode(sentence1)
# embedding2 = model.encode(sentence2)
# embedding3 = model.encode(sentence3)

# # 计算余弦相似度
# similarity = util.pytorch_cos_sim(embedding1, embedding2)
# similarity1 = util.pytorch_cos_sim(embedding3, embedding2)

# # 打印相似度值
# print(f"Similarity1: {similarity.item()}")
# print(f"Similarity2: {similarity1.item()}")