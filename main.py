from app.interface import create_app

app = create_app(
    model_path     = 'models/qa_model_v2.pth',
    tokenizer_path = 'models/tokenizer.json'
)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)