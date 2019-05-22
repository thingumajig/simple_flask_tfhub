#!flask/bin/python
import os

from flask import Flask, jsonify, request, make_response, abort, g
from tfhub_context import TFHubContext, ElmoTFHubContext
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

app = Flask(__name__)
embedding_context = {}


def get_ec(ctype) -> TFHubContext:
  global embedding_context
  ec = embedding_context.get(ctype, None)
  if ec is None:
    print('Creating new TFHubContext!')
    if ctype == 'elmo':
      ec = ElmoTFHubContext(type='default')
    else:
      if ctype == 'use':
        ec = TFHubContext()
      else:
        abort(400)
    embedding_context[ctype] = ec

  return embedding_context[ctype]

# @app.teardown_appcontext
# def teardown_ec(error):
#   ec = g.pop('ec ', None)
#
#   if ec is not None:
#     ec.close()
#
#   print('Teardown embedding_context')

@app.route('/use/api/v1.0/hello', methods=['POST','GET'])
def hello():
  return jsonify({'version': '1.0'})


@app.route('/use/api/v1.0/sentence/<ctype>', methods=['POST','GET'])
def get_sentence_emb(ctype):
  sentence = None

  if request.args:
    sentence = request.args.get('sentence', None)

  if not sentence and request.form:
    try:
      sentence = request.form['sentence']
    except:
      sentence = None

  if not sentence and request.json:
    sentence = request.json.get('sentence', None)


  if not sentence:
    abort(400)

  emb_tensor = get_ec(ctype).get_embedding([sentence])[0].tolist()

  return jsonify({'embedings': emb_tensor})

@app.route('/use/api/v1.0/text/<ctype>', methods=['POST','GET'])
def get_text_emb(ctype):
  text = None

  if request.args:
    text = request.args.get('text', None)

  if not text and request.form:
    try:
      text = request.form['text']
    except:
      text = None

  if not text and request.json:
    text = request.json.get('text', None)

  if not text:
    abort(400)

  # sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
  # sentences = sent_tokenizer.tokenize(text)
  sentences = sent_tokenize(text)

  emb_tensor = get_ec(ctype).get_embedding(sentences)

  json = {}
  for i in range(0, len(sentences)):
    json[f'sentence_{i}'] = {'sentence': sentences[i], 'embedding': emb_tensor[i].tolist()}

  return jsonify(json)



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found', 'errorCode': str(error)}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'BadRequest', 'errorCode': str(error)}), 400)


if __name__ == '__main__':
  app.run(
    host=os.getenv('LISTEN', '0.0.0.0'),
    port=int(os.getenv('PORT', '8080')),
    debug=True)
