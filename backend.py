from teller import Teller
from flask import Flask, request, jsonify
from flask_cors import CORS

flask_app = Flask(__name__)
CORS(flask_app)

models = {
    'shakespeare': Teller('shakespeare_model.pt'),
    'comte': Teller('comte.pt'),
    'poe': Teller('edgar_allan_poe.pt'),
    'kjbible': Teller('kjbible_model.pt')
}

@flask_app.route('/suggest', methods=['POST'])
def suggest():
    print('request received')
    posted_data = request.get_json()
    print(posted_data)
    text = posted_data['text']
    corpus = posted_data['corpus']

    if corpus not in models:
        print(f'corpus {corpus} not found')
        return jsonify({'error': 'corpus not found'})
    else:
        suggestion = models[corpus].predict_next_word(text, 1)
        return jsonify({'suggestions': suggestion})


    

if __name__ == '__main__':
    print('Started running')
    flask_app.run(host='0.0.0.0', port=8000, debug=True)