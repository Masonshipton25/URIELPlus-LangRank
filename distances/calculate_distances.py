# Calculates linguistic distances between language pairs from datasets (dep, el, mt, pos) using URIEL+, outputs results to CSV files.

# Import URIEL+ for calculating language distances and the csv module for writing output files
from urielplus import urielplus as uriel
import pandas as pd
import csv

# Initialize the URIEL+ system and enable caching for efficiency
u = uriel.URIELPlus()
u.reset()
u.set_cache(True)
u.integrate_databases()        # Integrate all linguistic sources
u.softimpute_imputation()      # Fill missing values using soft 


# Dictionary mapping the language codes (2-letter) from dep.csv to their corresponding ISO 639-3 (3-digit) codes
ISO_639_3_MAPPING = {
    "cs": "ces",  # Czech
    "ru": "rus",  # Russian
    "no": "nor",  # Norwegian
    "es": "spa",  # Spanish
    "ko": "kor",  # Korean
    "et": "ekk",  # Estonian
    "pl": "pol",  # Polish
    "nl": "nld",  # Dutch
    "pt": "por",  # Portuguese
    "la": "lat",  # Latin
    "fr": "fra",  # French
    "de": "deu",  # German
    "hi": "hin",  # Hindi
    "ca": "cat",  # Catalan
    "it": "ita",  # Italian
    "en": "eng",  # English
    "fi": "fin",  # Finnish
    "bg": "bul",  # Bulgarian
    "sl": "slv",  # Slovenian
    "sk": "slk",  # Slovak
    "ro": "ron",  # Romanian
    "hr": "hrv",  # Croatian
    "ar": "ara",  # Arabic
    "lv": "lav",  # Latvian
    "he": "heb",  # Hebrew
    "uk": "ukr",  # Ukrainian
    "id": "ind",  # Indonesian
    "da": "dan",  # Danish
    "sv": "swe",  # Swedish
    "zh": "zho",  # Chinese (Mandarin)
    "kk": "kaz",  # Kazakh
    "hy": "hye",  # Armenian
    "lt": "lit",  # Lithuanian
    "be": "bel",  # Belarusian
    "mr": "mar",  # Marathi
    "ta": "tam",  # Tamil
    "ga": "gle",  # Irish
    "hu": "hun",  # Hungarian
    "te": "tel",  # Telugu
    "af": "afr",  # Afrikaans
    "vi": "vie",  # Vietnamese
    "ug": "uig",  # Uighur
    "el": "ell",  # Greek
    "gl": "glg",  # Galician
    "sr": "srp",  # Serbian
    "tr": "tur",  # Turkish
    "ur": "urd",  # Urdu
    "cu": "chu",  # Church Slavik
    "fa": "pes",  # Persian
    "eu": "eus",  # Basque
    "ja": "jpn",  # Japanese
    "am": "amh",  # Amharic
    "th": "tha",  # Thai
    "yo": "yor",  # Yoruba
    "tl": "tgl",  # Tagalog
    "br": "bre",  # Breton
    "sa": "san",  # Sanskrit
    "fo": "fao",  # Faroese
}

# Dictionary mapping the ISO 639-3 codes in URIEL but without equivalent glottocodes in URIEL+, and the glottocodes that were used as replacements
MANUAL_CODE_FIXES = {
    "alb": "alba1267",  # Albanian
    "ara": "stan1318",  # Arabic
    "aze": "nort2697",  # Azerbaijani
    "zho": "mand1415",  # Chinese (Mandarin)
    "est": "esto1258",  # Estonian
    "msa": "stan1306",  # Malay
    "orm": "east2652",  # Oromo
    "fas": "west2369",  # Persian
    "swa": "swah1253"   # Swahili
}

map_df = pd.read_csv('distances//urielplus//database//urielplus_csvs//uriel_glottocode_map.csv')
iso_to_glottocode = dict(zip(map_df['code'], map_df['glottocode']))
iso_to_glottocode.update(MANUAL_CODE_FIXES)

# List of linguistic distance types to calculate between language pairs
DISTANCES = ["genetic", "syntactic", "featural", "phonological", "inventory", "geographic"]


# Calculate distances for languages in dep.
dep_df = pd.read_csv("experiment_csvs//URIEL//dep.csv")

dep_lang_pairs = dep_df[['Target lang', 'Transfer lang']].values.tolist()

iso_dep_lang_pairs = [[ISO_639_3_MAPPING.get(lang1, lang1), ISO_639_3_MAPPING.get(lang2, lang2)] for lang1, lang2 in dep_lang_pairs]

glotto_dep_lang_pairs = [[iso_to_glottocode.get(lang1, lang1), iso_to_glottocode.get(lang2, lang2)] for lang1, lang2 in iso_dep_lang_pairs]

# Open CSV file to write dep distances
with open("distances//dep_distances.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    # Write the header row to the CSV
    header = ["Target lang", "Transfer lang"] + [d.upper() for d in DISTANCES]
    writer.writerow(header)

    # Loop through each language pair and compute distances
    for lang_pair in glotto_dep_lang_pairs:
        distance_values = u.new_distance(DISTANCES, lang_pair)  # Returns list of distances

        # Print formatted distances to console (for debugging/inspection)
        distance_str = ",".join(f"{value:.4f}" for value in distance_values)
        print(distance_str)

        # Write result to CSV: [source_lang, target_lang, distances...]
        writer.writerow([lang_pair[0], lang_pair[1]] + distance_values)


# Calculate distances for languages in el.csv
el_df = pd.read_csv('experiment_csvs//URIEL//el.csv')

el_lang_pairs = el_df[['Target lang', 'Transfer lang']].values.tolist()

glotto_el_lang_pairs = [[iso_to_glottocode.get(lang1, lang1), iso_to_glottocode.get(lang2, lang2)] for lang1, lang2 in el_lang_pairs]

# Open CSV file to write el distances
with open("distances//el_distances.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    # Write the header row to the CSV
    header = ["Target lang", "Transfer lang"] + [d.upper() for d in DISTANCES]
    writer.writerow(header)

    # Loop through each language pair and compute distances
    for lang_pair in glotto_el_lang_pairs:
        distance_values = u.new_distance(DISTANCES, lang_pair)  # Returns list of distances

        # Print formatted distances to console (for debugging/inspection)
        distance_str = ",".join(f"{value:.4f}" for value in distance_values)
        print(distance_str)

        # Write result to CSV: [source_lang, target_lang, distances...]
        writer.writerow([lang_pair[0], lang_pair[1]] + distance_values)


# Calculate distances for languages in mt.
mt_df = pd.read_csv("experiment_csvs//URIEL//mt.csv")

mt_lang_pairs = mt_df[['Source lang' , 'Transfer lang']].values.tolist()

glotto_mt_lang_pairs = [[iso_to_glottocode.get(lang1, lang1), iso_to_glottocode.get(lang2, lang2)] for lang1, lang2 in mt_lang_pairs]

# Open CSV file to write mt distances
with open("distances//mt_distances.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    # Write the header row to the CSV
    header = ["Source lang", "Transfer lang"] + [d.upper() for d in DISTANCES]
    writer.writerow(header)

    # Loop through each language pair and compute distances
    for lang_pair in glotto_mt_lang_pairs:
        distance_values = u.new_distance(DISTANCES, lang_pair)  # Returns list of distances

        # Print formatted distances to console (for debugging/inspection)
        distance_str = ",".join(f"{value:.4f}" for value in distance_values)
        print(distance_str)

        # Write result to CSV: [source_lang, target_lang, distances...]
        writer.writerow([lang_pair[0], lang_pair[1]] + distance_values)


# Calculate distances for languages in pos.
pos_df = pd.read_csv("experiment_csvs//URIEL//pos.csv")

pos_lang_pairs = pos_df[['Task lang' , 'Aux lang']].values.tolist()

iso_pos_lang_pairs = [[ISO_639_3_MAPPING.get(lang1, lang1), ISO_639_3_MAPPING.get(lang2, lang2)] for lang1, lang2 in pos_lang_pairs]

glotto_pos_lang_pairs = [[iso_to_glottocode.get(lang1, lang1), iso_to_glottocode.get(lang2, lang2)] for lang1, lang2 in iso_pos_lang_pairs]

# Open CSV file to write pos distances
with open("distances//pos_distances.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    # Write the header row to the CSV
    header = ["Task lang" , "Aux lang"] + [d.upper() for d in DISTANCES]
    writer.writerow(header)

    # Loop through each language pair and compute distances
    for lang_pair in glotto_pos_lang_pairs:
        distance_values = u.new_distance(DISTANCES, lang_pair)  # Returns list of distances

        # Print formatted distances to console (for debugging/inspection)
        distance_str = ",".join(f"{value:.4f}" for value in distance_values)
        print(distance_str)

        # Write result to CSV: [source_lang, target_lang, distances...]
        writer.writerow([lang_pair[0], lang_pair[1]] + distance_values)