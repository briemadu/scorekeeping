#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2022 Madureira, Brielen
# SPDX-License-Identifier: MIT

"""
This module contains two functions.
- propositions_from_caption: generate propositions (entailments and 
  contradicitons) from a VisDial caption.
- generate_proposition: given a q/a pair, check which rule can be used and 
  return propositions (entailment and contradiction).
"""

import inflect
import spacy

import lists
import rules
from rules import noun_is_singular, noun_is_plural
from rules import answer_is_unsure

nlp = spacy.load("en_core_web_sm")
inflect = inflect.engine()


def propositions_from_caption(caption):
    """Special rule to generate propositions from captions."""
    props = []
    doc_caption = nlp(caption)
    for token in doc_caption:
        # existence of all mentioned nouns
        if token.pos_ == 'NOUN' and token.head.pos_ != 'NOUN':
            obj = token.text
            if noun_is_plural(obj):
                entailment = 'there are {}.'.format(obj)
                contradiction = 'there are no {}.'.format(obj)
            else:
                entailment = 'there is {}.'.format(inflect.a(obj))
                contradiction = 'there is no {}.'.format(obj)
            props.append(((entailment, contradiction), None))

        # characteristics, adjectives followed by nouns
        elif token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
            attr = token.text
            obj = token.head.text
            if noun_is_singular(obj):
                entailment = 'one can see {} {}.'.format(
                                                    inflect.a(attr), obj)
                contradiction = 'one cannot see {} {}.'.format(
                                                    inflect.a(attr), obj)
            else:
                entailment = 'one can see {} {}.'.format(attr, obj)
                contradiction = 'one cannot see {} {}.'.format(attr, obj)
            props.append(((entailment, contradiction), None))
    return props


def generate_proposition(q, a, coref_q, coref_a):
    """Given a QA pair, check if it matches a rule; if yes, return propositions."""
    if answer_is_unsure(a):
        return []
    # LEXICAL RULES
    # what color is the dog / what color is her hair
    if q.startswith(lists.what_color_questions):
        return rules.what_color(q, a, coref_q), 'what_color'
    # what color boat (but ignore 'what color' with no NP)
    if q.startswith('what color') and len(q.split()) > 2:
        return rules.what_color(q, a, coref_q, no_verb=True), 'what_color'
    # can you see [anything as long as you want also with pronoun]
    if q.startswith(lists.can_see_questions):
        return rules.can_you_see(q, a, coref_q), 'can_you_see'
    # are there [anything as long as you want also with pronoun]
    if q.startswith(lists.there_questions):
        return rules.there(q, a, coref_q), 'there'
    # any dog
    if q.startswith(lists.any_questions):
        return rules.any_(q, a, coref_q), 'any_'
    # what kind of dog? old
    if q.startswith(lists.what_kind_questions) and len(a.split()) < 3:
        return rules.what_kind(q, a), 'what_kind'
    # is it sunny
    if q in lists.sunny_questions:
        return rules.sunny(q, a), 'sunny'
    # is it cloudy
    if q in lists.cloudy_questions:
        return rules.cloudy(q, a), 'cloudy'
    # is it daytime
    if q in lists.daytime_questions:
        return rules.daytime(q, a), 'daytime'
    # is it a black dog
    if q.startswith(lists.is_it_questions):
        return rules.is_it(q, a), 'is_it'
    # what is the weather like
    if q.startswith(lists.weather_questions):
        return rules.weather(q, a), 'weather'
    # is it inside / outside
    if q in lists.in_out_questions:
        return rules.is_inside(q, a), 'is_inside'
    # inside
    if a in lists.inside_answers:
        return rules.in_answer(a), 'in_answer'
    # outdoors
    if a in lists.outside_answers:
        return rules.out_answer(a), 'out_answer'
    # is the photo in color
    if q in lists.photo_color_questions:
        return rules.image_in_color(q, a), 'image_in_color'
    # daytime as answer
    if a in lists.daytime_answers:
        return rules.daytime_answer(a), 'daytime_answer'
    # sunny as answer
    if a in lists.weather_answers:
        return rules.weather_answer(a), 'sunny/cloudy'
    # male as female in answer
    if a in lists.person_answers:
        return rules.person_answer(a), 'male/female'
    # no people as answer
    if a.startswith(lists.no_people_answers):
        return rules.no_people(a), 'no_people'

    # GRAMMATICAL RULES

    # PROPN makes no difference for us, so treat is as PROP
    pos = [token.pos_ if token.pos_ != 'PROPN' else 'NOUN' for token in nlp(q)]
    pos = " ".join(pos)
    doc_a = nlp(a)
    # are the wheels (super) big
    if pos in lists.is_noun_adj_questions:
        return rules.is_noun_adj(q, a), 'is_noun_adj'
    if pos in lists.noun_questions:
        return rules.noun_(q, a), 'noun_'
    # is the cat on the mat
    if pos in lists.noun_prep_det_noun_questions:
        return rules.noun_prep_noun(q, a), 'noun_prep_noun'
    # is the cat on mat
    if pos in lists.noun_prep_noun_questions:
        return rules.noun_prep_noun(q, a, det=False), 'noun_prep_noun'
    # is the cat sleeping
    if pos in lists.noun_verb_questions:
        return rules.noun_verb(q, a), 'noun_verb'
    # is the boy wearing a hat
    # does the man have a bag
    if pos in lists.noun_verb_det_noun_questions:
        # "AUX DET NOUN VERB NOUN"
        return rules.noun_verb_noun(q, a), 'noun_verb_noun'
    # is the boy playing football
    # does the market sell food
    if pos in lists.noun_verb_noun_questions:
        return rules.noun_verb_noun(q, a, det=False), 'noun_verb_noun'
    # does the plane has [anything]
    if (pos.startswith(('AUX DET NOUN VERB','AUX DET NOUN AUX')) 
        and q.split()[3] == 'have'):
        return rules.have(q, a), 'have'
    # does the baby look/seem cute
    if pos in lists.look_adj_questions:
        return rules.look_adj(q, a, coref_q), 'look_adj'
    # are the people workers
    if pos in lists.noun_noun_questions:
        return rules.noun_noun(q, a), 'noun_noun'
    # what is the man wearing / doing (ignore other verbs)
    if pos in lists.what_is_questions:
        return rules.what_is(q, a), 'what_is'
    # is the dog black or white
    if pos in lists.adj_or_adj_questions:
        return rules.adj_or_adj(q, a), 'adj_or_adj'
    # is the person a policeman
    # is the baby a boy or a girl
    # is the flower a red flower
    if pos.startswith('AUX DET NOUN DET'):
        return rules.is_np_np(q, a), 'is_np_np'
    # answer is a single noun, therefore it should exist
    if len(doc_a) == 1 and doc_a[0].pos_ in ('NOUN',):
        return rules.noun_(a, 'yes'), 'noun_'
    # does it look like a person
    # do they look like a family
    if pos in lists.look_questions and q.split()[3] == 'like':
        return rules.look_like(q, a, coref_q), 'look_like'
    if 'PRON' in pos and coref_q != q:
        # is he old
        if pos in lists.verb_pron_adj_questions:
            return rules.verb_pron_adj(q, a, coref_q), 'verb_pron_adj'
        # is he holding the cup
        # also fits do you see, but it was handled before
        if pos in lists.verb_pron_det_obj_questions:
            return rules.verb_pron_obj(q, a, coref_q), 'verb_pron_obj'
        # is she playing football
        # also fits do you see, but it was handled before
        if pos in lists.verb_pron_obj_questions:
            return rules.verb_pron_obj(q, a, coref_q, det=False), 'verb_pron_obj'
        # is she sleeping
        if pos in lists.pron_verb_questions:
            return rules.noun_verb(q, a, coref_q), 'noun_verb'
        # are they on the boat
        if pos in lists.pron_prep_questions:
            return rules.pron_prep(q, a, coref_q), 'pron_prep'
        # what is she holding
        if pos in lists.what_is_pron_questions:
            return rules.what_is(q, a, coref_q), 'what_is'

    return None
