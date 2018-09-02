def evaluate(encoder, decoder, in_lang, max_length=MAX_LENGTH):
    if use_cuda:
        in_lang = in_lang.cuda()
    input_variable = Variable(in_lang)
    input_variable = input_variable.unsqueeze(0)
    input_length = input_variable.size(1)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(torch.tensor([input_variable[0][ei]]), encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = int(topi[0][0])
        if ni == EOS_token:
            break
        else:
            decoded_words.append(lang_dataset.output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair_idx = random.choice(list(range(len(lang_dataset))))
        pair = lang_dataset.pairs[pair_idx]
        in_lang, out_lang = lang_dataset[pair_idx]
        output_words = evaluate(encoder, decoder, in_lang)
        output_sentence = ' '.join(output_words)
        print('Input : ', pair[0], ' | Desired Output : ', pair[1], ' | Generated Output : ', output_sentence)
