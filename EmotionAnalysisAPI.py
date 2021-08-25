#import libraries
from Business.TextPredication import TwitterReterival
from Business.TextPredication import textPredictor
#-------------------------------------------
#API liberaries
from flask import Flask,request
from flask_cors import CORS
import jsonpickle
app = Flask(__name__)
CORS(app)
#-------------------------------------------
UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#-----------------------------------------------
@app.route('/getEmotionFromTwitter', methods=["GET"])
def getEmotionFromTwitter():    
    term = request.args['term']     
    tweets=TwitterReterival.get_tweetsbyKeyword(term)
    output=textPredictor.predictTweets(tweets)   
    json_string=jsonpickle.encode(output,unpicklable=False)
    response = app.response_class(
        response=json_string,
        status=200,
        mimetype='application/json' )
    return response


#-----------------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000,debug=False, threaded=False)
    
