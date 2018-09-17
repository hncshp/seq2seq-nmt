# seq2seq encoder_decoer model, soft and hard attention, batched beam search, attention heat map

1. replace "./data/xxx" with your data files
2. replace "\<unk\>" and "\</s\>" with your preferred characters
3. model supports '\<space\>' splited and no '\<space\>' splited sentences, so please choose the right value for "FLAGS.whitespace_or_nonws_slip", True means '\<space\>' splited, False means no '\<space\>' splited
4. "FLAGS.fix_or_variable_length" define fixed or variable length mode.
