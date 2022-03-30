#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
This module contains auxiliary lists of words / phrases / POS TAG patterns.
"""

PATH_ALLENNLP = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"

# _________________________________  ANSWERS _________________________________

positive_answers = (
    # common
    'yes', 'yes it is', 'i think so', 'yes they are', 'looks like it',
    'yes, it is', 'yes he is', 'yes it does', 'yeah', 'yes there is',
    'yes i can', 'it is', 'yes 1',
    # others
    'correct', 'looks like it', 'it looks like it', 'yup', 'yes it is',
    'yes she is', 'yes there are', 'yes, it is', 'yes, he is', 'yes, she is',
    'yes, they are', 'yes, it does', 'yes, there is', 'yes I can', 'right'
    )

negative_answers = (
    # common
    'no', 'not that i can see', 'nope', 'not really', 'i don\'t think so',
    'i don\'t see any', 'no i can\'t', 'not at all', 'not that i see',
    'nothing', '0', 'i do not see any',
    # others
    'can\'t see any', 'i don\'t see', 'nope', 'i don\'t see it',
    'no, i can\'t', 'i don’t see', 'i don’t think so', 'i don’t see any',
    'no i can’t', 'can’t see any', 'i don’t see it', 'no, i can’t'
    )

unsure_answers = (
    # common
    'can\'t tell', 'i can\'t tell', 'not sure', 'maybe', 'i cannot tell',
    'i don\'t know', 'can\'t see', 'don\'t know', 'hard to tell',
    'cannot tell', 'not visible', 'i\'m not sure',
    # others
    'unknown', 'not sure', 'i can\'t see', 'i cannot see', 'i can not see',
    'can not see', 'can not tell', 'no way', 'i’m not sure', 'i can’t see',
    'can’t tell', 'i can’t tell', 'i don’t know', 'can’t see', 'don’t know',
    )

colors = (
    # common
    'white', 'black', 'brown', 'blue', 'red', 'green', 'gray', 'yellow',
    'silver', 'grey', 'tan', 'beige', 'orange', 'pink',
    # others
    'violet', 'gold', 'purple', 'transparent', 'blonde', 'metal', 'blond',
    'stainless', 'wood',
    )

numbers = (
    # common
    '2', '1', '0', '3', '4', '5', '6', '0 that i can see', 'just 1', '7', '8',
    # others
    '9', '10',
    )

quantities = (
    'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'
    'lots', 'a few', 'many', 'a lot', 'hundreds',
    )

daytime_answers = ('daytime', 'daytime,', 'day', 'day time')

inside_answers = ('inside', 'indoors')

no_people_answers = ('no people',)

outside_answers = ('outside', 'outdoors')

person_answers = ('male', 'female', 'man', 'woman')

weather_answers = ('sunny', 'cloudy', 'overcast')


# ________________________________ QUESTIONS _________________________________

adj_or_adj_questions = (
    'AUX DET NOUN ADJ CCONJ ADJ',
    )

any_questions = (
    'any ',
    )

can_see_questions = (
    'can you see', 'do you see',
    )

cloudy_questions = (
    'is it cloudy', 'is it cloudy out', 'is it cloudy outside',
    'is it a cloudy day',
    )

daytime_questions = (
    'is it daytime', 'is it day time'
     )

how_many_questions = (
    'ADV ADJ NOUN', 'ADV ADJ NOUN AUX ADV'
    )

in_out_questions = (
    'is it inside', 'is it outside', 'is it indoors', 'is this indoors'
    'is it outdoors', 'is this inside', 'is this outside', 'is this outdoors'
    )

is_it_questions = (
    'is it a',
    )

is_noun_adj_questions = (
    'AUX NOUN ADJ', 'AUX DET NOUN ADJ', 'AUX DET NOUN ADV ADJ',
    'AUX DET NOUN ADV',
    )

look_questions = (
    'AUX PRON VERB ADP DET NOUN', 'AUX PRON VERB ADP NOUN',
    'AUX PRON VERB ADP DET ADJ', 'AUX PRON VERB ADP ADJ',
    'AUX PRON VERB ADP DET NOUN NOUN', 'AUX PRON VERB ADP DET ADJ NOUN'
    )

look_adj_questions = (
    'AUX DET NOUN VERB ADJ', 'AUX PRON VERB ADJ'
    )

noun_questions = (
    'NOUN',
    )

noun_noun_questions = (
    'AUX DET NOUN NOUN',
    )

noun_prep_det_noun_questions = (
    'AUX DET NOUN ADP DET NOUN',
    )

noun_prep_noun_questions = (
    'AUX DET NOUN ADP NOUN',
    )

noun_verb_questions = (
    'AUX DET NOUN VERB',
    )

noun_verb_det_noun_questions = (
    'AUX DET NOUN VERB DET NOUN',
    )

noun_verb_noun_questions = (
    'AUX DET NOUN VERB NOUN',
    )

photo_color_questions = (
    'is this in color', 'is the photo in color', 'is this photo in color',
    'is the picture in color', 'is this picture in color',
    'is the image in color', 'is this image in color',
    )

pron_prep_questions = (
    'AUX PRON ADP DET NOUN',
    )

pron_verb_questions = (
    'AUX PRON VERB',
    )

sunny_questions = (
    'is it sunny', 'is it sunny out', 'is it sunny outside',
    'does it appear to be a sunny day', 'is it a sunny day',
    )

there_questions = (
    'are there any', 'is there any', 'is there a ', 'is there an ',
    'are there', 'is there'
    )

there_prep_questions = (
    'AUX PRON DET NOUN ADP DET NOUN',
    )

verb_pron_adj_questions = (
    'AUX PRON ADJ',
    )

verb_pron_det_obj_questions = (
    'AUX PRON VERB DET NOUN',
    )

verb_pron_obj_questions = (
    'AUX PRON VERB NOUN',
    )

what_color_questions = (
    'what color is', 'what colour is', 'what color are', 'what colour are'
    )

what_is_questions = (
    'PRON AUX DET NOUN VERB',
    )

what_is_pron_questions = (
    'PRON AUX PRON VERB',
    )

what_kind_questions = (
    'what kind of', 'what type of'
    )

weather_questions = (
    'what is the weather like', 'what is weather like' 'what\'s weather like',
    'what\'s the weather like', 'what is the weather',
    'how is the weather', 'how is weather',  'how\'s the weather',
    'what does the weather look like', 'what kind of weather is it',
    )

# __________________________________  OTHERS _________________________________

pronouns_to_solve = set(('it', 'he', 'she', 'they', 'this', 'that',
                         'these', 'those'))

oblique_pronouns_to_solve = set(('him', 'them'))

possessive_pronouns_to_solve = set(('its', 'his', 'her', 'hers',
                                    'their', 'theirs'))

coref_pronouns = pronouns_to_solve.union(oblique_pronouns_to_solve).union(
    possessive_pronouns_to_solve)

# _________________________________  FILTERS _________________________________

# VisDial dialogues sometimes contain words that may be innappropriate or
# offensive or used for profanity when a offends q.

filter_1 = set(['playboy', 'playboys',
                'devilishly', 'devilishlies',
                'kill', 'kills',
                'spirit', 'spirits',
                'hookah', 'hookahs',
                'boob', 'boobs',
                'drunk', 'drunks',
                'daredevil', 'daredevils',
                'fart', 'farts',
                'orgy', 'orgies',
                'hoar', 'hoars',
                'steamy', 'steamies',
                'fags', 'fag',
                'psychic', 'psychics',
                'weenie', 'weenies',
                'shitt', 'shitts',
                'snatch', 'snatches',
                'len',
                'slave', 'slaves',
                'seamen', 'seamens',
                'dildo', 'dildoes',
                'pantys', 'panty',
                'screw', 'screws',
                'demon', 'demons',
                'anti-gay', 'anti-gays',
                'cum', 'cums',
                'dick', 'dicks',
                'lmao', 'lmaos',
                'cowgirl', 'cowgirls',
                'negro', 'negroes',
                'spik', 'spiks',
                'puss', 'pusses',
                'nappy', 'nappies',
                'pedo', 'pedoes',
                'race', 'races',
                'wad', 'wads',
                'shag', 'shags',
                'tard', 'tards',
                'muff', 'muffs',
                'organ', 'organs',
                'jap', 'japs',
                'sperm', 'sperms',
                'cocks', 'cock',
                'raped', 'rapeds',
                'slut', 'sluts',
                'asses', 'ass',
                'ugly', 'uglies',
                'busty', 'busties',
                'aryan', 'aryans',
                'jewish', 'jewishes',
                'homey', 'homeys',
                'wanky', 'wankies',
                'lesbians', 'lesbian',
                'wank', 'wanks',
                'loin', 'loins',
                'panty', 'panties',
                'gai', 'gais',
                'bullshit', 'bullshits',
                'fat', 'fats',
                'piss', 'pisses',
                'devil', 'devils',
                'seaman', 'seamen',
                'boobs', 'boob',
                'pissing', 'pissings',
                'religious', 'religiouses',
                'toots', 'toot',
                'sucked', 'suckeds',
                'stoned', 'stoneds',
                'fag', 'fags',
                'bewitched', 'bewitcheds',
                'cumming', 'cummings',
                'prig', 'prigs',
                'xx', 'xxes',
                'shite', 'shites',
                'tits', 'tit',
                'pimp', 'pimps',
                'shitting', 'shittings',
                'pasty', 'pasties',
                'teste', 'testes',
                'gay', 'gays',
                'stupid', 'stupids',
                'teets', 'teet',
                'vulgar', 'vulgars',
                'twat', 'twats',
                'screwed', 'screweds',
                'hooker', 'hookers',
                'spac', 'spacs',
                'breasts', 'breast',
                'brown shower', 'brown showers',
                'hemp', 'hemps',
                'pron', 'prons',
                'coon', 'coons',
                'hiv', 'hivs',
                'snuff', 'snuffs',
                'nipple', 'nipples',
                'hell', 'hells',
                'dumbass', 'dumbasses',
                'frigg', 'friggs',
                'moron', 'morons',
                'pee', 'pees',
                'bastard', 'bastards',
                'humped', 'humpeds',
                'nude', 'nudes',
                'dummy', 'dummies',
                'weirdo', 'weirdos',
                'pervert', 'perverts',
                'raper', 'rapers',
                'sissy', 'sissies',
                'whitey', 'whiteys',
                'jackass', 'jackasses',
                'hore', 'hores',
                'klan', 'klans',
                'tramp', 'tramps',
                'queer', 'queers',
                'underwear', 'underwears',
                'revue', 'revues',
                'nob', 'nobs',
                'dong', 'dongs',
                'phallic', 'phallics',
                'crotch', 'crotches',
                'omg', 'omgs',
                'peepee', 'peepees',
                'wedgie', 'wedgies',
                'erect', 'erects',
                'lust', 'lusts',
                'halfnaked', 'halfnakeds',
                'stroke', 'strokes',
                'urinal', 'urinals',
                'undies', 'undy',
                'killer', 'killers',
                'fuckin', 'fuckins',
                'suck', 'sucks',
                'bra', 'bras',
                'panties', 'panty',
                'humping', 'humpings',
                'pornography', 'pornographies',
                'racy', 'racies',
                'kinky', 'kinkies',
                'godlike', 'godlikes',
                'prostitute', 'prostitutes',
                'woody', 'woodies',
                'gringo', 'gringos',
                'vomit', 'vomits',
                'vodka', 'vodkas',
                'unholy', 'unholies',
                'douche', 'douches',
                'xxx', 'xxxes',
                'voyeur', 'voyeurs',
                'nig', 'nigs',
                'wang', 'wangs',
                'jerk', 'jerks',
                'oral', 'orals',
                'scantily', 'scantilies',
                'kraut', 'krauts',
                'teat', 'teats',
                'bdsm', 'bdsms',
                'murder', 'murders',
                'gays', 'gay',
                'virgin', 'virgins',
                'pecker', 'peckers',
                'duche', 'duches',
                'trashy', 'trashies',
                'spic', 'spics',
                'hump', 'humps',
                'chink', 'chinks',
                'cock', 'cocks',
                'pissin', 'pissins',
                'damn', 'damns',
                'skank', 'skanks',
                'nipples', 'nipple',
                'nudes', 'nude',
                'pedophile', 'pedophiles',
                'knob', 'knobs',
                'phalli', 'phallis',
                'annoying', 'annoyings',
                'nazi', 'nazis',
                'strip club', 'strip clubs',
                'pantie', 'panties',
                'nad', 'nads',
                'threesome', 'threesomes',
                'gey', 'geys',
                'hooters', 'hooter',
                'kum', 'kums',
                'fanny', 'fannies',
                'scumbag', 'scumbags',
                'poon', 'poons',
                'she male', 'she males',
                'cowgirls', 'cowgirl',
                'lingerie', 'lingeries',
                'hoer', 'hoers',
                'erotic', 'erotics',
                'penis', 'penises',
                'whore', 'whores',
                'antigay', 'antigays',
                'pastie', 'pasties',
                'meth', 'meths',
                'gayish', 'gayishes',
                'anus', 'anuses',
                'pissed', 'pisseds',
                'sleazy', 'sleazies',
                'semen', 'semens',
                'titi', 'titis',
                'hooter', 'hooters',
                'scum', 'scums',
                'arse', 'arses',
                'gae', 'gaes',
                'gaytime', 'gaytimes',
                'stfu', 'stfus',
                'anal', 'anals',
                'ass', 'asses',
                'spick', 'spicks',
                'horny', 'hornies',
                'rum', 'rums',
                'dyke', 'dykes',
                'tit', 'tits',
                'shits', 'shit',
                'rimming', 'rimmings',
                'potty', 'potties',
                'maxi', 'maxis',
                'shitless', 'shitlesses',
                'weed', 'weeds',
                'fucking', 'fuckings',
                'thug', 'thugs',
                'blue waffle', 'blue waffles',
                'pussy', 'pussies',
                'shemale', 'shemales',
                'pantyhose', 'pantyhoses',
                'vag', 'vags',
                'whiz', 'whizzes',
                'naked', 'nakeds',
                'rape', 'rapes',
                'paki', 'pakis',
                'massa', 'massas',
                'cipa', 'cipas',
                'lech', 'leches',
                'half-naked', 'half-nakeds',
                'sexy', 'sexies',
                'terd', 'terds',
                'fisting', 'fistings',
                'sucking', 'suckings',
                'bondage', 'bondages',
                'arian', 'arians',
                'porno', 'pornoes',
                'prick', 'pricks',
                'sexual', 'sexuals',
                'pawn', 'pawns',
                'reich', 'reiches',
                'junky', 'junkies',
                'shitty', 'shitties',
                'homo', 'homoes',
                'molest', 'molests',
                'lube', 'lubes',
                'bitch', 'bitches',
                'rump', 'rumps',
                'demonic', 'demonics',
                'turd', 'turds',
                'fuck', 'fucks',
                'vagina', 'vaginas',
                'hitler', 'hitlers',
                'crap', 'craps',
                'rottencrap', 'rottencraps',
                'porn', 'porns',
                'sandbar', 'sandbars',
                'cunt', 'cunts',
                'thrust', 'thrusts',
                'facial', 'facials',
                'retard', 'retards',
                'sniper', 'snipers',
                'shit', 'shits',
                'poop', 'poops',
                'raping', 'rapings',
                'weiner', 'weiners',
                'sex', 'sexes',
                'witch', 'witches',
                'urine', 'urines',
                'tampon', 'tampons',
              ])

# a few others used in interjections
filter_2 = set(['jesus', 'holy', 'gods', 'christ', 'god'])
