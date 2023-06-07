import pandas as pd
from urllib.parse import urlparse
import ipaddress
import re

def getDomain(url):
  domain = urlparse(url).netloc
  if re.match(r"^www.",domain):
    domain = domain.replace("www.","")
  return domain

def havingIP(url):
  try:
    ipaddress.ip_address(url)
    ip = 1
  except:
    ip = 0
  return ip

def haveAtSign(url):
  if "@" in url:
    at = 1    
  else:
    at = 0    
  return at

def getLength(url):
  if len(url) < 54:
    length = 0            
  else:
    length = 1            
  return length

def getDepth(url):
  s = urlparse(url).path.split('/')
  depth = 0
  for j in range(len(s)):
    if len(s[j]) != 0:
      depth = depth+1
  return depth

def redirection(url):
  pos = url.rfind('//')
  if pos > 6:
    if pos > 7:
      return 1
    else:
      return 0
  else:
    return 0

def httpDomain(url):
  domain = urlparse(url).netloc
  if 'https' in domain:
    return 1
  else:
    return 0

shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"
                      
def tinyURL(url):
    match=re.search(shortening_services,url)
    if match:
        return 1
    else:
        return 0
    
def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1            # phishing
    else:
        return 0    
    
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime



def domainAge(domain_name):
  creation_date = domain_name.creation_date
  expiration_date = domain_name.expiration_date
  if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
    try:
      creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if ((expiration_date is None) or (creation_date is None)):
      return 1
  elif ((type(expiration_date) is list) or (type(creation_date) is list)):
      return 1
  else:
    ageofdomain = abs((expiration_date - creation_date).days)
    print(ageofdomain)
    if ((ageofdomain/30) < 6):
      age = 1
    else:
      age = 0
  return age


def domainEnd(domain_name):
  expiration_date = domain_name.expiration_date
  if isinstance(expiration_date,str):
    try:
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if (expiration_date is None):
      return 1
  elif (type(expiration_date) is list):
      return 1
  else:
    today = datetime.now()
    end = abs((expiration_date - today).days)
    if ((end/30) < 6):
      end = 1
    else:
      end = 0
  return end

import requests

def iframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"[<iframe>|<frameBorder>]", response.text):
          return 0
      else:
          return 1
      
def mouseOver(response): 
  if response == "" :
    return 1
  else:
    if re.findall("<script>.+onmouseover.+</script>", response.text):
      return 1
    else:
      return 0
  
    
def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1

    
def forwarding(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1
    
def featureExtraction(url):

  features = []
  #Address bar based features (10)
  features.append(getDomain(url))
  features.append(havingIP(url))
  features.append(haveAtSign(url))
  features.append(getLength(url))
  features.append(getDepth(url))
  features.append(redirection(url))
  features.append(httpDomain(url))
  features.append(tinyURL(url))
  features.append(prefixSuffix(url))
  
  #Domain based features (4)
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1

  features.append(dns)
  features.append(1 if dns == 1 else domainAge(domain_name))
  features.append(1 if dns == 1 else domainEnd(domain_name))
  
  # HTML & Javascript based features (4)
  try:
    response = requests.get(url)
  except:
    response = ""
  features.append(iframe(response))
  features.append(mouseOver(response))
  features.append(rightClick(response))
  features.append(forwarding(response))
  
  return features

from xgboost import XGBClassifier
#from sklearn.metrics import accuracy_score

def outp(url):
    legi_features = []
    legi_features.append(featureExtraction(url))
    feature_names = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                          'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
                          'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
    legitimate = pd.DataFrame(legi_features, columns= feature_names)
    legitimate = legitimate.drop(['Domain'], axis = 1).copy()
    data0 = pd.read_csv('C:/Users/vedav/Desktop/project/urldata.csv')
    data = data0.drop(['Domain'], axis = 1).copy()
    data = data.sample(frac=1).reset_index(drop=True)
    y = data['Label']
    X = data.drop('Label',axis=1)
    xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
    xgb.fit(X, y)
    y_test_xgb = xgb.predict(legitimate)
    if y_test_xgb[0]==0:
        return "THIS IS A SAFE URL"+"\N{grinning face with smiling eyes}"
    else:
        return "BEWARE! IT'S A MALICIOUS URL"+"\N{skull and crossbones}"
    
from sklearn.neural_network import MLPClassifier
    
def output2(url):
    legi_features = []
    legi_features.append(featureExtraction(url))
    feature_names = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                          'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
                          'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards']
    legitimate = pd.DataFrame(legi_features, columns= feature_names)
    legitimate = legitimate.drop(['Domain'], axis = 1).copy()
    data0 = pd.read_csv('C:/Users/vedav/Desktop/project/urldata.csv')
    data = data0.drop(['Domain'], axis = 1).copy()
    data = data.sample(frac=1).reset_index(drop=True)
    y = data['Label']
    X = data.drop('Label',axis=1)
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))
    mlp.fit(X, y)

    y_test_xgb = mlp.predict(legitimate)
    if y_test_xgb[0]==0:
        return "THIS IS A SAFE URL"+"\N{grinning face with smiling eyes}"
    else:
        return "BEWARE! IT'S A MALICIOUS URL"+"\N{skull and crossbones}"

#web_framework
from flask import Flask,render_template,request
app=Flask(__name__)
@app.route("/")
@app.route("/home")

def home():
    return render_template("index.html")

@app.route("/result",methods=['POST','GET'])

def result():
    output=request.form.to_dict()
    url=output["name"]
    return render_template("index.html",name=outp(url))

if __name__=='__main__':
    app.run(debug=True,port=5007)








