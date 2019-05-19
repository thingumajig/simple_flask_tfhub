#!flask/bin/python
from flask import Flask, jsonify, request, make_response, abort, g

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tfhub_context import TFHubContext

app = Flask(__name__)
embedding_context = None


def get_ec() -> TFHubContext:
  global embedding_context
  if embedding_context is None:
    print('Creating new TFHubContext!')
    embedding_context = TFHubContext()

  return embedding_context

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


@app.route('/use/api/v1.0/use-sentence', methods=['POST','GET'])
def get_sentence_encoding():
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

  emb_tensor = get_ec().get_embedding([sentence])[0].tolist()

  return jsonify({'embedings': emb_tensor})



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found', 'errorCode': str(error)}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'BadRequest', 'errorCode': str(error)}), 400)

if __name__ == '__main__':
  app.run(debug=True, port=5000)