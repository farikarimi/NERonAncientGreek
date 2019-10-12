import csv
import eli5
from greek_accentuation.characters import base
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import cross_val_predict
import xml.etree.ElementTree as ET


# reading the gold standard lists of toponyms and ethnonyms
with open('data/places_gold-standard_updated.txt', 'r') as places_file:
    places_txt = places_file.read()

with open('data/ethnics_gold-standard_updated.txt', 'r') as ethnics_file:
    ethnics_txt = ethnics_file.read()

# with open('data/places_gold-standard.txt', 'r') as places_file:
#     places_txt = places_file.read()
#
# with open('data/ethnics_gold-standard.txt', 'r') as ethnics_file:
#     ethnics_txt = ethnics_file.read()


# removes all diacritics from a string
def strip_diacritics(s):
    return ''.join(base(c) for c in s)


# lists containing the toponyms/ethnonyms (w/o diacritics) from the gold standard lists
places_gold_list = [strip_diacritics(place) for place in places_txt.splitlines()]

ethnics_gold_list = [strip_diacritics(ethnic) for ethnic in ethnics_txt.splitlines()]

ethnics_and_places = places_gold_list + ethnics_gold_list


# returns the entity type for a given token based on the gold standard lists
def get_type(token):
    if token in places_gold_list:
        # label for toponyms
        return 'place'
    if token in ethnics_gold_list:
        # label for ethnonyms
        return 'ethnic'
    else:
        # label for any other word
        return '0'


# returns true if a sentence contains a token from one of the gold standard lists
def has_gold_word(sent):
    result = False
    if result is False:
        for token in sent:
            if strip_diacritics(token) in ethnics_and_places:
                result = True
    return result


# reading the XML-file with the annotated text of Histories
tree = ET.parse('data/hdt.xml')
root = tree.getroot()

sentences = []

# creating a list of sentences that contain a token from one of the gold standard lists
for div in root.find('source'):
    for sentence in div.findall('sentence'):
        forms = [token.get('form') for token in sentence.findall('token') if token.get('form')]
        if has_gold_word(forms):
            tokens = []
            for token in sentence.findall('token'):
                if token.get('form'):
                    # each token is a 3-tuple consisting of a form, a POS-tag and an entity type
                    token = (token.get('form'),
                             token.get('part-of-speech'),
                             get_type(strip_diacritics(token.get('form'))))
                    tokens.append(token)
            # each sentence consists of a list of tokens
            sentences.append(tokens)

# printing the number of sentences that contain a token from the gold standard lists
print('\n' + str(len(sentences)) + ' sentences\n\n')


# returns the list of features for a token
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'postag': postag,
        'postag[:2]': postag[:1]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:1]
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:1]
        })
    else:
        features['EOS'] = True

    return features


# returns a list containing the features for each token in a sentence
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# returns a list containing the labels (entity type) for each token in a sentence
def sent2labels(sent):
    return [label for token, pos, label in sent]


# returns a list containing the forms for each token in a sentence
def sent2tokens(sent):
    return [form for form, pos, label in sent]


# training the CRF model
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

crf = sklearn_crfsuite.CRF(c1=0.1,
                           c2=0.1,
                           max_iterations=100)

crf.fit(X, y)

# cross-validating the model and showing the weights assigned to each feature
pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
report = flat_classification_report(y_pred=pred, y_true=y)

print('flat_classification_report:\n\n', report, '\n\n')
print('cross_val_predict:\n\n', pred, '\n\n')
print('eli5.explain_weights(crf, top=100):\n', eli5.format_as_text(eli5.explain_weights(crf, top=100)))


# creates CSV-files containing all tokens to which the model has assigned a label that
# doesn't match the token's label in the training data
def perf_measure(sents, y_actual, y_hat):
    f1 = open('results/predicted_tokens_updated.csv', 'w')
    f2 = open('results/misclassified_tokens_updated.csv', 'w')
    # f1 = open('results/predicted_tokens.csv', 'w')
    # f2 = open('results/misclassified_tokens.csv', 'w')
    number1 = 0
    number2 = 0
    header = ['no', 'token', 'pos', 'actual_label', 'predicted_label', 'sent_no', 'token_no', 'sent']
    writer1 = csv.DictWriter(f1, fieldnames=header)
    writer1.writeheader()
    writer2 = csv.DictWriter(f2, fieldnames=header)
    writer2.writeheader()
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            if y_actual[i][j] != y_hat[i][j]:
                # predictions that could be right
                if y_actual[i][j] == '0':
                    sent = [sents[i][k][0] for k in range(len(sents[i]))]
                    number1 += 1
                    writer1.writerow({
                        'no': str(number1),
                        'token': sents[i][j][0],
                        'pos': sents[i][j][1],
                        'actual_label': sents[i][j][2],
                        'predicted_label': y_hat[i][j],
                        'sent_no': str(i),
                        'token_no': str(j),
                        'sent': ' '.join(sent)
                    })
                # misclassifications
                else:
                    number2 += 1
                    writer2.writerow({
                        'no': str(number2),
                        'token': sents[i][j][0],
                        'pos': sents[i][j][1],
                        'actual_label': sents[i][j][2],
                        'predicted_label': y_hat[i][j],
                        'sent_no': str(i),
                        'token_no': str(j),
                        'sent': ' '.join(sent)
                    })
    f1.close()
    f2.close()


perf_measure(sents=sentences, y_actual=y, y_hat=pred)
