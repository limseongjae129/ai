from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# 모델과 토크나이저 로드
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 사용하는 모델 입력
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.cuda()

def generate_response(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return "죄송합니다. 응답 생성 중 오류가 발생했습니다."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        response = generate_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': '오류가 발생했습니다. 다시 시도해주세요.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)