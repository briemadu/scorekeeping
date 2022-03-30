#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
Manipulation rules that turn QA pairs into propositions.
Some auxiliary functions for the rules.
"""

import inflect
import pattern.en
import random
import spacy

from lists import positive_answers, negative_answers, unsure_answers
from lists import colors


nlp = spacy.load("en_core_web_sm")
inflect = inflect.engine()


def answer_is_positive(a):
    if a in positive_answers or a.startswith(positive_answers):
        return True
    return False


def answer_is_negative(a):
    if a in negative_answers or a.startswith(negative_answers):
        return True
    return False


def answer_is_unsure(a):
    if a in unsure_answers or a.startswith(unsure_answers):
        return True
    return False


def noun_is_plural(noun):
    if inflect.singular_noun(noun):
        return True
    return False


def noun_is_singular(noun):
    if inflect.singular_noun(noun):
        return False
    return True


# these two functions are from Python 3.9
def removeprefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix):]
    return string[:]


def removesuffix(string, suffix):
    # suffix='' should not call self[:-0].
    if suffix and string.endswith(suffix):
        return string[:-len(suffix)]
    return string[:]


# _____________________________ MANIPULATION RULES __________________________


def what_color(q, a, coref, no_verb=False):
    """
    Manipulation rule for 'what color' QA pairs:
        what color is the door? white.
        -> the door is white / the door is not white
        what color is the car? looks like dark gray
        -> the car is dark gray / the car is not dark gray
    """
    # ignore some strange questions with no noun
    if q in ('what color is', 'what color are', 'what color is '):
        return []
    q = coref
    # Case 1: what color ball
    if no_verb:
        _, color, *np = q.split()
        verb = 'are' if inflect.singular_noun(np[0]) else 'is'
        if len(np) > 1:
            np = 'the ' + " ".join(np)
        else:
            np = 'the ' + np[0]
    # Case 2: what color is the ball
    else:
        _, color, verb, *np = q.split(' ')
        if np[0] != 'the':
            np = ['the'] + np
        np = " ".join(np)
    # get all colors in answer (assuming all are correct)
    words_in_answer = set([w.replace(',', '') for w in a.split()])
    np_colors = list(words_in_answer.intersection(colors))

    # retrieve dark/light back
    for c, color in enumerate(np_colors):
        if 'dark ' + color in a:
            np_colors[c] = 'dark ' + np_colors[c]
        if 'light ' + color in a:
            np_colors[c] = 'light ' + np_colors[c]

    # check how many colors were found
    if not np_colors:
        return []
    if len(np_colors) == 1:
        right_color = np_colors[0]
    elif len(np_colors) == 2:
        right_color = np_colors[0] + ' and ' + np_colors[1]
    else:
        right_color = ", ".join(np_colors[:-1]) + ' and ' + np_colors[-1]

    entailment = f'{np} {verb} {right_color}.'
    contradiction = f'{np} {verb} not {right_color}.'

    return (entailment, contradiction), None


def can_you_see(q, a, coref_q):
    """
    Manipulation rule for 'can you see' QA pairs:
        can you see the water? yes
        -> one can see the water / one cannot see any water
    """

    common_dets = ('other', 'anyone', 'anything')
    _, _, _, *subj = coref_q.split()

    # this type of question will make no sense as a proposition
    if subj in (['anything', 'else'], ['anyone', 'else']):
        return []
    if subj[0] in ('a', 'an', 'any', 'the'):
        _, *subj = subj

    det_subj = subj[:]
    if subj[0] not in common_dets and noun_is_singular(subj[0]):
        det_subj[0] = inflect.a(subj[0])
    if subj[0] in ('anyone', 'anything'):
        det_subj[0] = det_subj[0].replace('any', 'some')
        subj[0] = subj[0].replace('any', '')
    det_subj = " ".join(det_subj)
    subj = " ".join(subj)

    if answer_is_positive(a):
        entailment = f'one can see {det_subj.strip()}.'
        contradiction = f'one cannot see any {subj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'one cannot see any {subj}.'
        contradiction = f'one can see {det_subj.strip()}.'
        return (entailment, contradiction), 'negative'
    return []


def there(q, a, coref_q):
    """
    Manipulation rule for 'there is/are' QA pairs:
        are there potatoes? i do not see any
        -> there are no potatoes. / there are potatoes.
    """
    verb, _, *np = coref_q.split()
    if np[:2] in (['anything', 'else'], ['anyone', 'else']):
        return []
    if np[0] == 'any':
        np = np[1:]

    np_noun = np[:]
    if verb == 'is' and np[0] in ('a', 'an'):
        np_noun = np[1:]
    if np[0] in ('anyone', 'anything'):
        np_noun = np[:]
        np_noun[0] = np_noun[0].replace('any', '')
        np[0] = np[0].replace('any', 'some')
    np = " ".join(np)
    np_noun = " ".join(np_noun)

    if answer_is_positive(a):
        entailment = f'there {verb} {np}.'
        contradiction = f'there {verb} no {np_noun}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'there {verb} no {np_noun}.'
        contradiction = f'there {verb} {np}.'
        return (entailment, contradiction), 'negative'
    return []


def any_(q, a, coref_q):
    """
    Manipulation rule for 'any noun' QA pairs:
        any people? no
        -> there are no people. / there are people.
    """
    q = coref_q.replace('any ', '')
    noun = q.split()[0]
    if noun == 'other':
        return []
    det_q = None
    verb = 'are'
    if noun_is_singular(noun):
        verb = 'is'
        det_q = inflect.a(q)

    if answer_is_positive(a):
        entailment = f'there {verb} {det_q or q}.'
        contradiction = f'there {verb} no {q}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'there {verb} no {q}.'
        contradiction = f'there {verb} {det_q or q}.'
        return (entailment, contradiction), 'negative'
    return []


def weather(q, a):
    """
    Manipulation rule for 'what is the whether like' QA pairs:
        what is the weather like? sunny.
        -> the weather is sunny. / the weather is not sunny.
    """
    doc = nlp(a)
    adjs = [token.text for token in doc if token.pos_ == 'ADJ'
            if token.text not in colors]
    if 'overcast' in a and 'overcast' not in adjs:
        # because spacy does not classify this as ADJ
        adjs.append('overcast')
    if not adjs:
        return []
    if len(adjs) > 1:
        adjs = ", ".join(adjs[:-1]) + ' and ' + adjs[-1]
    else:
        adjs = adjs[0]

    entailment = f'the weather is {adjs}.'
    contradiction = f'the weather is not {adjs}.'
    return (entailment, contradiction), None


def what_kind(q, a):
    """
    Manipulation rule for 'what kind of noun' QA pairs:
        what kind of food? asian.
        -> the food is asian. / the food is not asian.

    only works for simple answers
    """
    words = q.split()
    # what kind/type of NOUN
    if len(words) == 4:
        noun = words[-1]
    # what kind of NOUN is this / are there / is it
    elif " ".join(words[-2:]) in (
            "are there", "is there", "is it", "is this", "is that", "is he",
            "is she", "are they"):
        noun = " ".join(words[3:-2])
    else:
        return []

    verb = 'is'
    if noun_is_plural(noun):
        verb = 'are'

    entailment = f'the {noun} {verb} {a}.'
    contradiction = f'the {noun} {verb} not {a}.'
    return (entailment, contradiction), None


def is_it(q, a):
    """
    Manipulation rule for 'is it a noun' QA pairs:
        is it a smartphone? yes.
        -> there is a smartphone. / there is no smartphone.
    """
    verb, _, det, *subj = q.split()
    subj = " ".join(subj)

    if answer_is_positive(a):
        entailment = f'there {verb} {det} {subj}.'
        contradiction = f'there {verb} not {det} {subj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'there {verb} not {det} {subj}.'
        contradiction = f'there {verb} {det} {subj}.'
        return (entailment, contradiction), 'negative'
    return []


def is_inside(q, a):
    """
    Manipulation rule for 'is this inside' QA pairs:
        is this outdoors? yes.
        > the image is outdoors. / the image is not outdoors.
    """
    aux, _, place = q.split()

    if answer_is_positive(a):
        entailment = f'the image {aux} {place}.'
        contradiction = f'the image {aux} not {place}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'the image {aux} not {place}.'
        contradiction = f'the image {aux} {place}.'
        return (entailment, contradiction), 'negative'
    return []


def in_answer(a):
    """
    Manipulation rule for answers 'inside' in QA pairs.
    """
    entailment = f'the image is {a}.'
    contradiction = f'the image is not {a}.'
    return (entailment, contradiction), None


def out_answer(a):
    """
    Manipulation rule for answers 'outside' in QA pairs.
    """
    entailment = f'the image is {a}.'
    contradiction = f'the image is not {a}.'
    return (entailment, contradiction), None


def sunny(q, a):
    """
    Manipulation rule for questions 'is it sunny' in QA pairs.
    """
    if answer_is_positive(a):
        entailment = 'it is sunny.'
        contradiction = 'it is not sunny.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = 'it is not sunny.'
        contradiction = 'it is sunny.'
        return (entailment, contradiction), 'negative'
    return []


def cloudy(q, a):
    """
    Manipulation rule for questions 'is it cloudy' in QA pairs.
    """
    if answer_is_positive(a):
        entailment = 'it is cloudy.'
        contradiction = 'it is not cloudy.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = 'it is not cloudy.'
        contradiction = 'it is cloudy.'
        return (entailment, contradiction), 'negative'
    return []


def weather_answer(a):
    """
    Manipulation rule for answers 'sunny/cloudy/overcast' in QA pairs.
    """
    entailment = f'it is {a}.'
    contradiction = f'it is not {a}.'
    return (entailment, contradiction), None


def daytime(q, a):
    """
    Manipulation rule for questions 'is it daytime' in QA pairs.
    """
    if answer_is_positive(a):
        entailment = 'it is daytime.'
        contradiction = 'it is not daytime.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = 'it is not daytime.'
        contradiction = 'it is daytime.'
        return (entailment, contradiction), 'negative'
    return []


def daytime_answer(a):
    """
    Manipulation rule for answers 'daytime' in QA pairs.
    """
    entailment = 'it is daytime.'
    contradiction = 'it is not daytime.'
    return (entailment, contradiction), None


def image_in_color(q, a):
    """
    Manipulation rule for questions about photo color in QA pairs.
    """
    noun = random.sample(['image', 'photo', 'picture'], 1)[0]
    if answer_is_positive(a):
        entailment = f'the {noun} is in color.'
        contradiction = f'the {noun} is not in color.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'the {noun} is not in color.'
        contradiction = f'the {noun} is in color.'
        return (entailment, contradiction), 'negative'
    return []


def person_answer(a):
    """
    Manipulation rule for answers male, female in QA pairs.
    """
    entailment = f'there is a {a}.'
    contradiction = f'there is no {a}.'
    return (entailment, contradiction), None


def no_people(a):
    """
    Manipulation rule for answer 'no people' QA pairs.
    """
    entailment = 'there are no people.'
    contradiction = 'there are people.'
    return (entailment, contradiction), None


def is_noun_adj(q, a):
    """
    Manipulation rule for 'is noun adj' QA pairs:
        are the wheels big? yes.
        -> the wheels are big. / the wheels are not big.
    """
    words = q.split()
    # Case 1: are hands clean
    if len(words) == 3:
        verb, noun, adj = words
        det = 'the'
    # Case 2: is the man tired / are the woman together
    elif len(words) == 4:
        verb, det, noun, adj = words
    #  Case 3: are the wheels very big
    elif len(words) == 5:
        verb, det, noun, adv, adj = words
        adj = adv + ' ' + adj
    if det == 'any':
        det = 'some'

    if answer_is_positive(a):
        entailment = f'{det} {noun} {verb} {adj}.'
        contradiction = f'{det} {noun} {verb} not {adj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{det} {noun} {verb} not {adj}.'
        contradiction = f'{det} {noun} {verb} {adj}.'
        return (entailment, contradiction), 'negative'
    return []


def noun_(q, a):
    """
    Manipulation rule for 'noun? yes' QA pairs:
        car? yes.
        -> there is a car. / there is no car.
    """
    if answer_is_positive(a):
        if noun_is_plural(q):
            entailment = f'there are {q}.'
            contradiction = f'there are no {q}.'
        else:
            entailment = f'there is {inflect.a(q)}.'
            contradiction = f'there is no {q}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        if noun_is_plural(q):
            entailment = f'there are no {q}.'
            contradiction = f'there are {q}.'
        else:
            entailment = f'there is no {q}.'
            contradiction = f'there is {inflect.a(q)}.'
        return (entailment, contradiction), 'negative'
    return []


def noun_prep_noun(q, a, det=True):
    """
    Manipulation rule for 'is the noun on the noun' QA pairs:
        is the cat on the sofa? yes.
        -> the cat is on the sofa. / the cat is not on the sofa.
    """
    if not det:
        verb, det, noun1, adp, noun2 = q.split()
    else:
        verb, det, noun1, adp, det2, noun2 = q.split()
        noun2 = det2 + ' ' + noun2

    if answer_is_positive(a):
        entailment = f'{det} {noun1} {verb} {adp} {noun2}.'
        contradiction = f'{det} {noun1} {verb} not {adp} {noun2}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{det} {noun1} {verb} not {adp} {noun2}.'
        contradiction = f'{det} {noun1} {verb} {adp} {noun2}.'
        return (entailment, contradiction), 'negative'
    return []


def noun_verb(q, a, coref_q=None):
    """
    Manipulation rule for 'is the noun verb-ing' QA pairs:
        is the dog sleeping? no.
        the dog is not sleeping. / the dog is sleeping.
    """
    if coref_q:
        aux, _, verb = q.split()
        noun = removesuffix(removeprefix(coref_q, aux + ' '), " " + verb)
        if noun in ('anyone', 'someone'):
            return []
    else:
        aux, det, noun, verb = q.split()
        noun = det + ' ' + noun

    if answer_is_positive(a):
        entailment = f'{noun} {aux} {verb}.'
        contradiction = f'{noun} {aux} not {verb}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{noun} {aux} not {verb}.'
        contradiction = f'{noun} {aux} {verb}.'
        return (entailment, contradiction), 'negative'
    return []


def noun_verb_noun(q, a, det=True):
    """
    Manipulation rule for 'is the noun verb-ing a noun' QA pairs:
        is the boy wearing a het? no.
        -> the boy is not wearing a hat. / the boy is wearing a hat.
    """
    if det:
        aux, det1, subj, verb, det2, obj = q.split()
        obj = det2 + ' ' + obj
    else:
        aux, det1, subj, verb, obj = q.split()
    if aux in ('is', 'are'):
        if answer_is_positive(a):
            entailment = f'{det1} {subj} {aux} {verb} {obj}.'
            contradiction = f'{det1} {subj} {aux} not {verb} {obj}.'
            return (entailment, contradiction), 'positive'
        if answer_is_negative(a):
            entailment = f'{det1} {subj} {aux} not {verb} {obj}.'
            contradiction = f'{det1} {subj} {aux} {verb} {obj}.'
            return (entailment, contradiction), 'negative'
    if aux == 'do':
        if answer_is_positive(a):
            entailment = f'{det1} {subj} {verb} {obj}.'
            contradiction = f'{det1} {subj} {aux} not {verb} {obj}.'
            return (entailment, contradiction), 'positive'
        if answer_is_negative(a):
            entailment = f'{det1} {subj} {aux} not {verb} {obj}.'
            contradiction = f'{det1} {subj} {verb} {obj}.'
            return (entailment, contradiction), 'negative'
    if aux == 'does':
        conj_verb = pattern.en.conjugate(verb, 'present', 3, 'singular')
        if answer_is_positive(a):
            entailment = f'{det1} {subj} {conj_verb} {obj}.'
            contradiction = f'{det1} {subj} {aux} not {verb} {obj}.'
            return (entailment, contradiction), 'positive'
        if answer_is_negative(a):
            entailment = f'{det1} {subj} {aux} not {verb} {obj}.'
            contradiction = f'{det1} {subj} {conj_verb} {obj}.'
            return (entailment, contradiction), 'negative'
    return []


def have(q, a):
    """
    Manipulation rule for 'does the noun have noun' QA pairs:
        does the horse have a tail? yes.
        -> the horse has a tail. / the horse does not have a tail.
    """
    aux, det, subj, verb, *obj = q.split()
    obj = " ".join(obj)
    verb2 = verb
    if aux == 'does':
        verb2 = 'has'

    if answer_is_positive(a):
        entailment = f'{det} {subj} {verb2} {obj}.'
        contradiction = f'{det} {subj} {aux} not {verb} {obj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{det} {subj} {aux} not {verb} {obj}.'
        contradiction = f'{det} {subj} {verb2} {obj}.'
        return (entailment, contradiction), 'negative'
    return []


def look_adj(q, a, coref_q):
    """
    Manipulation rule for 'does the noun look adj' QA pairs:
        does the baby seem cute? yes.
        -> the baby seems cute. / the baby does not seem cute.
    """
    aux, *noun, verb, adj = coref_q.split()
    noun = " ".join(noun)
    if verb not in ('look', 'seem'):
        return []
    if aux == 'does':
        conj_verb = pattern.en.conjugate(verb, 'present', 3, 'singular')
    else:
        conj_verb = verb

    if answer_is_positive(a):
        entailment = f'{noun} {conj_verb} {adj}.'
        contradiction = f'{noun} {aux} not {verb} {adj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{noun} {aux} not {verb} {adj}.'
        contradiction = f'{noun} {conj_verb} {adj}.'
        return (entailment, contradiction), 'negative'
    return []


def noun_noun(q, a):
    """
    Manipulation rule for 'are the noun noun' QA pairs:
        are the people workers? no.
        -> the people are not workers. / the people are workers.
    """
    aux, det, noun1, noun2 = q.split()

    if answer_is_positive(a):
        entailment = f'{det} {noun1} {aux} {noun2}.'
        contradiction = f'{det} {noun1} {aux} not {noun2}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{det} {noun1} {aux} not {noun2}.'
        contradiction = f'{det} {noun1} {aux} {noun2}.'
        return (entailment, contradiction), 'negative'
    return []


def what_is(q, a, coref_q=None):
    """
    Manipulation rule for 'what is the noun verb-ing' QA pairs:
        what is the woman doing? reading.
        -> the woman is reading. / the woman is not reading.
    """

    q = q.replace('what\'s', 'what is')

    if coref_q:
        coref_q = coref_q.replace('what\'s', 'what is')
        what, aux, _, verb = q.split()
        noun = removesuffix(removeprefix(coref_q, what + ' ' + aux + ' '),
                            " " + verb)
    else:
        what, aux, det, noun, verb = q.split()
        noun = det + ' ' + noun
    # ignore answers with pronouns because they're probably too complicated
    word1 = [token.pos_ for token in nlp(a)][0]
    if word1 not in ('VERB', 'NOUN', 'ADJ') and not a.startswith(('a ', 'an ')):
        return []
    if verb == 'wearing':
        entailment = f'{noun} {aux} {verb} {a}.'
        contradiction = f'{noun} {aux} not {verb} {a}.'
        return (entailment, contradiction), None
    if verb == 'doing' and a.split()[0].endswith('ing'):
        entailment = f'{noun} {aux} {a}.'
        contradiction = f'{noun} {aux} not {a}.'
        return (entailment, contradiction), None
    return []


def adj_or_adj(q, a):
    """
    Manipulation rule for 'is the noun adj or adj' QA pairs:
        is the smartphone new or old? new.
        -> the smartphone is new. / the smartphone is not new.
    """
    aux, det, noun, adj1, conj, adj2 = q.split()
    common = set([adj1, adj2]).intersection(set(a.split()))
    if conj == 'or':
        if common:
            adj = list(common)[0]
            entailment = f'{det} {noun} {aux} {adj}.'
            contradiction = f'{det} {noun} {aux} not {adj}.'
            return (entailment, contradiction), None
        return []
    return []


def is_np_np(q, a):
    """
    Manipulation rule for 'is the noun a noun' QA pairs:
        is the person a policeman? no.
        -> the person is not a policeman. / the person is a policeman.
    """
    aux, det, noun, *charac = q.split()
    charac = " ".join(charac)
    # is the baby a girl or a boy
    if 'or' in q and len(a.split()) == 1:
        entailment = f'{det} {noun} {aux} {inflect.a(a)}.'
        contradiction = f'{det} {noun} {aux} not {inflect.a(a)}.'
        return (entailment, contradiction), None
    if 'or' in q:
        return []
    if answer_is_positive(a):
        entailment = f'{det} {noun} {aux} {charac}.'
        contradiction = f'{det} {noun} {aux} not {charac}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{det} {noun} {aux} not {charac}.'
        contradiction = f'{det} {noun} {aux} {charac}.'
        return (entailment, contradiction), 'negative'
    return []


def verb_pron_adj(q, a, coref_q):
    """
    Manipulation rule for 'is pron adj' QA pairs:
        is he old? yes.
        [coref] is old. / [coref] is not old.
    """
    aux, _, adj = q.split()
    noun = removesuffix(removeprefix(coref_q, aux + ' '), " "+adj)

    if answer_is_positive(a):
        entailment = f'{noun} {aux} {adj}.'
        contradiction = f'{noun} {aux} not {adj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{noun} {aux} not {adj}.'
        contradiction = f'{noun} {aux} {adj}.'
        return (entailment, contradiction), 'negative'
    return []


def verb_pron_obj(q, a, coref_q, det=True):
    """
    Manipulation rule for 'is pron verb-ing the noun' QA pairs:
        is he holding the cup? no.
        -> [coref] is not holding the cup. / [coref] is holding the cup.
    """
    if det:
        aux, _, verb, det, obj = q.split()
        obj = det + ' ' + obj
    else:
        aux, _, verb, obj = q.split()
    subj = removesuffix(removeprefix(coref_q, aux + ' '),
                        " " + " ".join([verb, obj]))
    if answer_is_positive(a):
        entailment = f'{subj} {aux} {verb} {obj}.'
        contradiction = f'{subj} {aux} not {verb} {obj}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{subj} {aux} not {verb} {obj}.'
        contradiction = f'{subj} {aux} {verb} {obj}.'
        return (entailment, contradiction), 'negative'
    return []


def pron_prep(q, a, coref_q):
    """
    Manipulation rule for 'aux pron on the noun' QA pairs:
        are they on the boat? yes.
        -> [coref] are on the boat. / [coref] are not on the boat.
    """
    aux, _, adp, det, noun = q.split()
    subj = removesuffix(removeprefix(coref_q, aux + ' '),
                        " " + " ".join([adp, det, noun]))

    if answer_is_positive(a):
        entailment = f'{subj} {aux} {adp} {det} {noun}.'
        contradiction = f'{subj} {aux} not {adp} {det} {noun}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{subj} {aux} not {adp} {det} {noun}.'
        contradiction = f'{subj} {aux} {adp} {det} {noun}.'
        return (entailment, contradiction), 'negative'
    return []


def look_like(q, a, coref_q):
    """
    Manipulation rule for 'does it look like a noun' QA pairs:
        do they look like a family? yes.
        -> [coref] looks like a family. / [coref] does not look like a family.
    """
    aux, _, look, like, *sth = q.split()
    if look != 'look':
        return []
    sth = " ".join(sth)
    subj = removesuffix(removeprefix(coref_q, aux + ' '),
                        " " + " ".join([look, like, sth]))
    if subj != 'it' and q == coref_q:
        return []
    if aux == 'does':
        look = 'looks'

    if answer_is_positive(a):
        entailment = f'{subj} {look} like {sth}.'
        contradiction = f'{subj} {aux} not look like {sth}.'
        return (entailment, contradiction), 'positive'
    if answer_is_negative(a):
        entailment = f'{subj} {aux} not look like {sth}.'
        contradiction = f'{subj} {look} like {sth}.'
        return (entailment, contradiction), 'negative'
    return []
