# =====================================
# IMPORTS
# =====================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================
# DATASET
# =====================================
eng_texts = [
    "hello","hi","good morning","good evening","thank you",
    "how are you","i am fine","what is your name",
    "where are you from","nice to meet you",
    "good night","please help me",
    "what are you doing","i am studying",
    "do you speak telugu","yes i speak telugu"
]

tel_texts = [
    "start హలో end",
    "start హాయ్ end",
    "start శుభోదయం end",
    "start శుభ సాయంత్రం end",
    "start ధన్యవాదాలు end",
    "start మీరు ఎలా ఉన్నారు end",
    "start నేను బాగున్నాను end",
    "start మీ పేరు ఏమిటి end",
    "start మీరు ఎక్కడ నుండి వచ్చారు end",
    "start మిమ్మల్ని కలవడం ఆనందంగా ఉంది end",
    "start శుభ రాత్రి end",
    "start దయచేసి నాకు సహాయం చేయండి end",
    "start మీరు ఏమి చేస్తున్నారు end",
    "start నేను చదువుతున్నాను end",
    "start మీరు తెలుగు మాట్లాడగలరా end",
    "start అవును నేను తెలుగు మాట్లాడగలను end"
]

# =====================================
# TOKENIZATION
# =====================================
eng_tokenizer = Tokenizer()
tel_tokenizer = Tokenizer(filters='')

eng_tokenizer.fit_on_texts(eng_texts)
tel_tokenizer.fit_on_texts(tel_texts)

eng_seq = eng_tokenizer.texts_to_sequences(eng_texts)
tel_seq = tel_tokenizer.texts_to_sequences(tel_texts)

# =====================================
# PADDING
# =====================================
max_eng_len = max(len(i) for i in eng_seq)
max_tel_len = max(len(i) for i in tel_seq)

eng_seq = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
tel_seq = pad_sequences(tel_seq, maxlen=max_tel_len, padding='post')

# =====================================
# VOCAB SIZE
# =====================================
eng_vocab = len(eng_tokenizer.word_index) + 1
tel_vocab = len(tel_tokenizer.word_index) + 1

# =====================================
# MODEL
# =====================================
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(eng_vocab, 128)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_tel_len-1,))
dec_emb_layer = Embedding(tel_vocab, 128)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    dec_emb, initial_state=encoder_states)

decoder_dense = Dense(tel_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

# =====================================
# TRAIN DATA
# =====================================
decoder_input_data = tel_seq[:, :-1]
decoder_target_data = tel_seq[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# =====================================
# TRAIN
# =====================================
print("Training Started...")

model.fit(
    [eng_seq, decoder_input_data],
    decoder_target_data,
    batch_size=4,
    epochs=300,
    verbose=0
)

print("Training Completed")

# =====================================
# INFERENCE MODELS
# =====================================
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs)

decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# =====================================
# REVERSE INDEX
# =====================================
reverse_tel_index = dict(
    (i, word) for word, i in tel_tokenizer.word_index.items())

start_token = tel_tokenizer.word_index['start']
end_token = tel_tokenizer.word_index['end']

# =====================================
# TRANSLATE FUNCTION
# =====================================
def translate(sentence):

    sentence = sentence.lower()

    if sentence not in eng_texts:
        return "NOT FOUND"

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')

    states = encoder_model.predict(seq)

    target_seq = np.zeros((1,1))
    target_seq[0,0] = start_token

    decoded_sentence = ""

    for _ in range(max_tel_len):

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states)

        token = np.argmax(output_tokens[0, -1, :])

        if token == end_token:
            break

        word = reverse_tel_index.get(token, '')

        decoded_sentence += word + " "

        target_seq = np.zeros((1,1))
        target_seq[0,0] = token

        states = [h, c]

    return decoded_sentence.strip()

# =====================================
# USER INPUT
# =====================================
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    input_text = ""
    if request.method == 'POST':
        input_text = request.form.get('english_text', '').strip()
        if input_text:
            translation = translate(input_text)
    return render_template('index.html', translation=translation, input_text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)