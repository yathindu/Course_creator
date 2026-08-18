# Sinhala -> Romanized-Sinhala converter, required by SinhalaVITS (the model
# was trained on romanized input, not raw Sinhala script). Copied verbatim from
# dialoglk/SinhalaVITS-TTS-F1's romanizer.py (huggingface.co/dialoglk) -- not
# authored here, kept as its own file so the source stays traceable and this
# doesn't need a network fetch on every synthesis call.

import re

# -- Specials (vowels, diacritics, standalone signs) --

ro_specials = [
    ['ඓ', 'ai'],
    ['ඖ', 'au'],
    ['ඍ', 'ṛ'],
    ['ඎ', 'ṝ'],
    ['ඐ', 'ḹ'],
    ['අ', 'a'],
    ['ආ', 'ā'],
    ['ඇ', 'æ'], ['ඇ', 'Æ'],
    ['ඈ', 'ǣ'],
    ['ඉ', 'i'],
    ['ඊ', 'ī'],
    ['උ', 'u'],
    ['ඌ', 'ū'],
    ['එ', 'e'],
    ['ඒ', 'ē'],
    ['ඔ', 'o'],
    ['ඕ', 'ō'],
    ['ඞ්', 'ṅ'],
    ['ං', 'ṁ'], ['ං', 'ṃ'],
    ['ඃ', 'ḥ'], ['ඃ', 'Ḥ'],
]

# -- Consonants --

ro_consonants = [
    ['ඛ', 'kh'],
    ['ඨ', 'ṭh'],
    ['ඝ', 'gh'],
    ['ඡ', 'ch'],
    ['ඣ', 'jh'],
    ['ඦ', 'ñj'],
    ['ඪ', 'ḍh'],
    ['ඬ', 'ṇḍ'],
    ['ථ', 'th'],
    ['ධ', 'dh'],
    ['ඵ', 'ph'],
    ['භ', 'bh'],
    ['ඹ', 'mb'],
    ['ඳ', 'ṉd'],
    ['ඟ', 'ṉg'],
    ['ඥ', 'gn'],
    ['ක', 'k'],
    ['ග', 'g'],
    ['ච', 'c'],
    ['ජ', 'j'],
    ['ඤ', 'ñ'],
    ['ට', 'ṭ'],
    ['ඩ', 'ḍ'],
    ['ණ', 'ṇ'],
    ['ත', 't'],
    ['ද', 'd'],
    ['න', 'n'],
    ['ප', 'p'],
    ['බ', 'b'],
    ['ම', 'm'],
    ['ය', 'y'],
    ['ර', 'r'],
    ['ල', 'l'],
    ['ව', 'v'],
    ['ශ', 'ś'],
    ['ෂ', 'ş'], ['ෂ', 'ṣ'],
    ['ස', 's'],
    ['හ', 'h'],
    ['ළ', 'ḷ'],
    ['ෆ', 'f']
]

# -- Combinations (consonant + vowel signs) --

ro_combinations = [
    ['', '', '්'],
    ['', 'a', ''],
    ['', 'ā', 'ා'],
    ['', 'æ', 'ැ'],
    ['', 'ǣ', 'ෑ'],
    ['', 'i', 'ි'],
    ['', 'ī', 'ී'],
    ['', 'u', 'ු'],
    ['', 'ū', 'ූ'],
    ['', 'e', 'ෙ'],
    ['', 'ē', 'ේ'],
    ['', 'ai', 'ෛ'],
    ['', 'o', 'ො'],
    ['', 'ō', 'ෝ'],
    ['', 'ṛ', 'ෘ'],
    ['', 'ṝ', 'ෲ'],
    ['', 'au', 'ෞ'],
    ['', 'ḹ', 'ෳ']
]

# -- Generate consonant+vowel combos --

def create_conso_combi(combinations, consonants):
    conso_combi = []
    for combi in combinations:
        for conso in consonants:
            base_sinh = conso[0] + combi[2]
            base_rom = combi[0] + conso[1] + combi[1]
            conso_combi.append((base_sinh, base_rom))
    return conso_combi

ro_conso_combi = create_conso_combi(ro_combinations, ro_consonants)

# -- Core replace function --
def replace_all(text, mapping):

    # sort by length (to handle longest matches first)
    mapping = sorted(mapping, key=lambda x: len(x[0]), reverse=True)
    for sinh, rom in mapping:
        text = re.sub(sinh, rom, text)
    return text

# -- Main Sinhala -> Roman Function --
def sinhala_to_roman(text):

    # remove ZWJ (zero-width joiner)
    text = text.replace("‍", "")

    # do consonant+vowel combos first
    text = replace_all(text, ro_conso_combi)

    # then specials
    text = replace_all(text, ro_specials)
    return text
